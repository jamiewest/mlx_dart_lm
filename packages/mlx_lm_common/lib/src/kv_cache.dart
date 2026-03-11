import 'package:mlx_dart/mlx_dart.dart';

/// Interface for Key/Value caches used during autoregressive decoding.
///
/// Mirrors `KVCache` from mlx-swift-lm.
abstract interface class KVCache {
  int get offset;
  int? get maxSize;

  /// Append [keys] and [values] (shape `[batch, heads, seqLen, headDim]`)
  /// and return the full accumulated tensors.
  (MLXArray, MLXArray) update(MLXArray keys, MLXArray values);

  List<MLXArray> get state;
  set state(List<MLXArray> value);

  bool get isTrimmable;

  /// Trim [n] tokens from the end. Returns the actual number trimmed.
  int trim(int n);

  /// Returns an explicit mask for this cache state, or null when a simple
  /// causal mask is sufficient.
  MLXArray? makeMask(MLXContext ctx, {required int n, int? windowSize});

  String get cacheType;
}

/// Unbounded KV cache that grows by concatenating new entries.
///
/// Mirrors `KVCacheSimple` from mlx-swift-lm.
final class KVCacheSimple implements KVCache {
  KVCacheSimple();

  MLXArray? _keys;
  MLXArray? _values;

  @override
  int offset = 0;

  @override
  int? get maxSize => null;

  @override
  (MLXArray, MLXArray) update(MLXArray keys, MLXArray values) {
    if (_keys case final k?) {
      final ctx = keys.context;
      final newK = concatenate(ctx, [k, keys], axis: 2);
      final newV = concatenate(ctx, [_values!, values], axis: 2);
      k.dispose();
      _values!.dispose();
      _keys = newK;
      _values = newV;
    } else {
      _keys = keys;
      _values = values;
    }
    offset += keys.dim(2);
    return (_keys!, _values!);
  }

  @override
  List<MLXArray> get state =>
      [if (_keys != null) _keys!, if (_values != null) _values!];

  @override
  set state(List<MLXArray> value) {
    if (value.length == 2) {
      _keys = value[0];
      _values = value[1];
      offset = _keys!.dim(2);
    }
  }

  @override
  bool get isTrimmable => true;

  @override
  int trim(int n) {
    if (_keys == null) return 0;
    final available = _keys!.dim(2);
    final actual = n.clamp(0, available);
    if (actual <= 0) return 0;

    final rank = _keys!.ndim;
    final newLen = available - actual;
    final start = List.filled(rank, 0);
    final stop = List<int>.generate(rank, (d) => d == 2 ? newLen : _keys!.shape[d]);

    final newK = _keys!.slice(start: start, stop: stop);
    final newV = _values!.slice(start: start, stop: stop);
    _keys!.dispose();
    _values!.dispose();
    _keys = newK;
    _values = newV;
    offset -= actual;
    return actual;
  }

  @override
  MLXArray? makeMask(MLXContext ctx, {required int n, int? windowSize}) {
    if (n == 1) return null;
    return createCausalMask(ctx, n: n, offset: offset, windowSize: windowSize);
  }

  @override
  String get cacheType => 'simple';

  void dispose() {
    _keys?.dispose();
    _values?.dispose();
    _keys = null;
    _values = null;
  }
}

/// Fixed-size KV cache; when full, new entries overwrite the oldest ones
/// (preserving the first [keep] tokens as permanent context).
///
/// Mirrors `RotatingKVCache` from mlx-swift-lm.
final class RotatingKVCache implements KVCache {
  RotatingKVCache({required int maxSize, this.keep = 4}) : _maxSize = maxSize;

  final int _maxSize;
  final int keep;

  MLXArray? _keys;
  MLXArray? _values;

  @override
  int offset = 0;

  @override
  int? get maxSize => _maxSize;

  @override
  (MLXArray, MLXArray) update(MLXArray keys, MLXArray values) {
    final ctx = keys.context;
    final newTokens = keys.dim(2);

    if (_keys == null) {
      _keys = keys;
      _values = values;
      offset += newTokens;
      return (_keys!, _values!);
    }

    final currentLen = _keys!.dim(2);

    if (currentLen + newTokens <= _maxSize) {
      final newK = concatenate(ctx, [_keys!, keys], axis: 2);
      final newV = concatenate(ctx, [_values!, values], axis: 2);
      _keys!.dispose();
      _values!.dispose();
      _keys = newK;
      _values = newV;
      offset += newTokens;
      return (_keys!, _values!);
    }

    // Need to evict from the rotating region [keep, currentLen) to fit newTokens.
    final rank = _keys!.ndim;
    final kShape = _keys!.shape;

    // How many old rotating tokens to keep after eviction.
    final rotateKeep = (_maxSize - keep - newTokens).clamp(0, currentLen - keep);

    // Slice permanent prefix [0, keep).
    final prefixStop = List<int>.from(kShape);
    prefixStop[2] = keep;
    final kPrefix = _keys!.slice(start: List.filled(rank, 0), stop: prefixStop);
    final vPrefix = _values!.slice(start: List.filled(rank, 0), stop: prefixStop);

    MLXArray newK;
    MLXArray newV;
    if (rotateKeep > 0) {
      // Slice the tail of the rotating region [currentLen-rotateKeep, currentLen).
      final tailStart = List.filled(rank, 0);
      tailStart[2] = currentLen - rotateKeep;
      final kTail = _keys!.slice(start: tailStart, stop: List<int>.from(kShape));
      final vTail = _values!.slice(start: tailStart, stop: List<int>.from(kShape));
      newK = concatenate(ctx, [kPrefix, kTail, keys], axis: 2);
      newV = concatenate(ctx, [vPrefix, vTail, values], axis: 2);
      kTail.dispose();
      vTail.dispose();
    } else {
      newK = concatenate(ctx, [kPrefix, keys], axis: 2);
      newV = concatenate(ctx, [vPrefix, values], axis: 2);
    }
    kPrefix.dispose();
    vPrefix.dispose();
    _keys!.dispose();
    _values!.dispose();
    _keys = newK;
    _values = newV;
    offset += newTokens;
    return (_keys!, _values!);
  }

  @override
  List<MLXArray> get state =>
      [if (_keys != null) _keys!, if (_values != null) _values!];

  @override
  set state(List<MLXArray> value) {
    if (value.length == 2) {
      _keys = value[0];
      _values = value[1];
      offset = _keys!.dim(2);
    }
  }

  @override
  bool get isTrimmable => false;

  @override
  int trim(int n) => 0;

  @override
  MLXArray? makeMask(MLXContext ctx, {required int n, int? windowSize}) {
    if (n == 1) return null;
    final effectiveOffset = offset.clamp(0, _maxSize - 1);
    return createCausalMask(ctx,
        n: n, offset: effectiveOffset, windowSize: windowSize ?? _maxSize);
  }

  @override
  String get cacheType => 'rotating';

  void dispose() {
    _keys?.dispose();
    _values?.dispose();
    _keys = null;
    _values = null;
  }
}

// ---------------------------------------------------------------------------
// ArraysCache
// ---------------------------------------------------------------------------

/// Cache that holds an arbitrary list of state arrays — used by SSM layers.
///
/// Unlike [KVCacheSimple]/[RotatingKVCache] this does not store keys/values;
/// it stores opaque state tensors whose layout is model-defined.
///
/// Mirrors `ArraysCache` from mlx-swift-lm.
class ArraysCache implements KVCache {
  ArraysCache({int size = 0}) : _cache = List.filled(size, null);

  final List<MLXArray?> _cache;

  // KVCache bookkeeping — not meaningful for SSMs, kept for protocol compliance.
  @override
  int offset = 0;

  @override
  int? get maxSize => null;

  @override
  (MLXArray, MLXArray) update(MLXArray keys, MLXArray values) =>
      throw UnsupportedError('ArraysCache does not support KV update; use [] indexing');

  /// Read a cached state array by index.
  MLXArray? operator [](int index) => _cache[index];

  /// Write a state array at [index].
  void operator []=(int index, MLXArray? value) => _cache[index] = value;

  @override
  List<MLXArray> get state => _cache.whereType<MLXArray>().toList();

  @override
  set state(List<MLXArray> value) {
    for (var i = 0; i < value.length && i < _cache.length; i++) {
      _cache[i] = value[i];
    }
  }

  @override
  bool get isTrimmable => false;

  @override
  int trim(int n) => 0;

  @override
  MLXArray? makeMask(MLXContext ctx, {required int n, int? windowSize}) => null;

  @override
  String get cacheType => 'arrays';
}

// ---------------------------------------------------------------------------
// MambaCache
// ---------------------------------------------------------------------------

/// Specialised [ArraysCache] for Mamba/SSM layers with exactly two state slots:
/// conv state (index 0) and SSM state (index 1).
///
/// Mirrors `MambaCache` from mlx-swift-lm.
final class MambaCache extends ArraysCache {
  MambaCache() : super(size: 2);

  /// Convolutional state tensor.
  MLXArray? get convState => this[0];
  set convState(MLXArray? v) => this[0] = v;

  /// SSM recurrent state tensor.
  MLXArray? get ssmState => this[1];
  set ssmState(MLXArray? v) => this[1] = v;

  @override
  String get cacheType => 'mamba';
}
