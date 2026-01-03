# Raytracer Optimization Plan
Based on profiling analysis (200x200 image, AnotherScene)

## Summary of Bottlenecks

**Total runtime breakdown:**
- `cast_shadow_rays`: 3248.3s (76.8% in `is_ray_blocked` → `find_closest_intersection`)
- `find_closest_intersection`: 1725.1s (66.3% in `intersect_sphere` - 1144s)
- `calculate_phong_lighting`: 3273.7s (99.6% calling `cast_shadow_rays`)
- `trace_ray`: 3305.44s (99.1% calling `calculate_phong_lighting`)

**Key inefficiencies:**
1. Repeated `np.array()` conversions on same values (light.position, sphere.position, etc.)
2. Multiple `normalize()` calls using `np.linalg.norm()` (expensive)
3. No caching of pre-computed values
4. Loop overhead in `cast_shadow_rays` and `find_closest_intersection`

---

## Priority 1: Eliminate Redundant `np.array()` Conversions (EASY, HIGH IMPACT)

### Problem
- `light.position` converted to `np.array()` 314,121 times in `cast_shadow_rays` (line 215, 230)
- `sphere.position` converted 77.5M times in `intersect_sphere` (line 105, 121)
- `material.diffuse_color`, `material.specular_color`, `light.color` converted repeatedly
- `camera.position`, `camera.look_at` converted in every ray generation

**Estimated savings: 10-15% speedup**

### Implementation Details

1. **Convert objects to numpy arrays at initialization** (in `parse_scene_file` or `render_scene`):
   ```python
   # In render_scene, before the main loop:
   for light in lights:
       light.position = np.array(light.position, dtype=np.float64)
       light.color = np.array(light.color, dtype=np.float64)
   
   for surface in surfaces:
       if hasattr(surface, 'position'):
           surface.position = np.array(surface.position, dtype=np.float64)
   
   camera.position = np.array(camera.position, dtype=np.float64)
   camera.look_at = np.array(camera.look_at, dtype=np.float64)
   camera.up_vector = np.array(camera.up_vector, dtype=np.float64)
   
   for material in materials:
       material.diffuse_color = np.array(material.diffuse_color, dtype=np.float64)
       material.specular_color = np.array(material.specular_color, dtype=np.float64)
       material.reflection_color = np.array(material.reflection_color, dtype=np.float64)
   ```

2. **Modify class definitions** (if you want permanent solution):
   - In `light.py`, `surfaces/sphere.py`, `surfaces/cube.py`, `material.py`, `camera.py`
   - Convert to numpy arrays in `__init__` methods
   - **OR** use properties that cache the numpy array

3. **Remove `np.array()` calls in hot paths**:
   - Line 105: `oc = ray_origin - sphere.position` (already numpy)
   - Line 121: `normal = normalize(point - sphere.position)`
   - Line 215: `light_to_surface = normalize(intersection_point - light.position)`
   - Line 230: `light_sample = light.position + light.radius * (u * right + v * up)`
   - Line 247: `light_dir = light.position - intersection_point`
   - Line 258: `diffuse = ... * material.diffuse_color * light.color`

---

## Priority 2: Optimize `normalize()` Function (EASY, MEDIUM IMPACT)

### Problem
- `normalize()` uses `np.linalg.norm()` which computes full L2 norm (expensive)
- Called 5M+ times in shadow rays (line 231, 223)
- Called for every light calculation (line 111, 113)

**Estimated savings: 5-10% speedup**

### Implementation Details

**Option A: Use squared norm and divide directly (FASTEST)**
```python
def normalize(v):
    # v is already numpy array
    norm_sq = np.dot(v, v)  # Faster than np.linalg.norm
    if norm_sq < EPSILON * EPSILON:  # Check squared epsilon
        return v
    return v / np.sqrt(norm_sq)  # Single sqrt instead of norm()
```

**Option B: Use numpy's optimized normalize (if available)**
```python
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    # Use in-place division if possible
    return np.divide(v, norm)  # Sometimes faster than v / norm
```

**Option C: Vectorized normalization (for batches)**
For `cast_shadow_rays`, compute all directions first, then normalize in batch:
```python
# In cast_shadow_rays, replace lines 230-231:
light_samples = np.array([light.position + light.radius * ((i + np.random.random()) / n - 0.5) * right + 
                                    ((j + np.random.random()) / n - 0.5) * up 
                                    for i in range(n) for j in range(n)])
shadow_ray_dirs = light_samples - intersection_point
# Normalize all at once (more efficient)
norms = np.linalg.norm(shadow_ray_dirs, axis=1, keepdims=True)
shadow_ray_dirs = np.divide(shadow_ray_dirs, norms, where=(norms > EPSILON))
```

**Recommendation: Start with Option A (easiest, good speedup)**

---

## Priority 3: Optimize `cast_shadow_rays` - Reduce Array Allocations (MEDIUM, HIGH IMPACT)

### Problem
- Line 230: `np.array(light.position) + ...` called 5M times
- Line 227-228: `np.random.random()` called individually (2 calls per iteration)
- Line 231: `normalize()` called 5M times
- Line 232: `np.linalg.norm()` called 5M times

**Estimated savings: 10-20% speedup**

### Implementation Details

1. **Pre-allocate random values** (if acceptable to use same seed pattern):
   ```python
   # Before the loops, generate all random values at once:
   n = int(scene_settings.root_number_shadow_rays)
   total_rays = n * n
   
   # Generate all random offsets at once
   random_offsets = np.random.random((n, n, 2))  # Shape: (n, n, 2) for u and v
   
   for i in range(n):
       for j in range(n):
           u = (i + random_offsets[i, j, 0]) / n - 0.5
           v = (j + random_offsets[i, j, 1]) / n - 0.5
           # ... rest of code
   ```

2. **Pre-compute coordinate system once** (already done, but verify it's outside loops)

3. **Vectorize light sample calculation**:
   ```python
   # Generate all (i, j) pairs
   i_indices, j_indices = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
   i_flat = i_indices.flatten()
   j_flat = j_indices.flatten()
   
   # Generate all random values
   random_u = np.random.random(n * n)
   random_v = np.random.random(n * n)
   
   u = (i_flat + random_u) / n - 0.5
   v = (j_flat + random_v) / n - 0.5
   
   # Vectorized computation
   light_samples = light.position + light.radius * (u[:, None] * right + v[:, None] * up)
   shadow_ray_dirs = light_samples - intersection_point
   ```

4. **Combine distance and normalization**:
   ```python
   # Instead of:
   shadow_ray_dir = normalize(light_sample - intersection_point)
   light_distance = np.linalg.norm(light_sample - intersection_point)
   
   # Do:
   shadow_ray_vec = light_sample - intersection_point
   light_distance = np.linalg.norm(shadow_ray_vec)
   shadow_ray_dir = shadow_ray_vec / light_distance  # Avoid normalize() overhead
   ```

**Recommendation: Start with #4 (easiest), then try #1 (good balance)**

---

## Priority 4: Optimize `intersect_sphere` (MEDIUM, HIGH IMPACT)

### Problem
- Called 77.5M times, taking 1144s (66.3% of `find_closest_intersection`)
- Line 105: `oc = ray_origin - sphere.position` (if position not pre-converted)
- Line 121: `normalize(point - sphere.position)` can be optimized
- Quadratic formula has redundant calculations

**Estimated savings: 5-15% speedup**

### Implementation Details

1. **Pre-convert sphere.position** (Priority 1)

2. **Optimize normal calculation**:
   ```python
   # Current (line 121):
   normal = normalize(point - sphere.position)
   
   # Optimized (if sphere.position is numpy):
   normal_vec = point - sphere.position
   # Since it's a sphere, we know the distance equals radius
   # So norm = radius, can skip normalize:
   normal = normal_vec / sphere.radius  # Faster!
   ```

3. **Early rejection with bounding sphere test** (if implemented):
   - Quick distance check before full intersection

4. **Optimize quadratic formula**:
   ```python
   # Current:
   a = np.dot(ray_direction, ray_direction)  # Always 1 if normalized!
   
   # If ray_direction is normalized (which it should be):
   a = 1.0  # Skip dot product
   oc = ray_origin - sphere.position
   b = 2.0 * np.dot(oc, ray_direction)
   c = np.dot(oc, oc) - sphere.radius * sphere.radius
   ```

**Note**: Verify ray_direction is always normalized (check `generate_ray`)

---

## Priority 5: Cache Repeated Calculations (EASY, MEDIUM IMPACT)

### Problem
- Line 251: `view_dir = normalize(-ray_direction)` computed for every light
- Material colors converted to arrays repeatedly
- `any(np.array(material.reflection_color) > 0)` computed every time

**Estimated savings: 2-5% speedup**

### Implementation Details

1. **Pre-compute view_dir once per intersection**:
   ```python
   # In calculate_phong_lighting, move outside light loop:
   view_dir = normalize(-ray_direction)  # Compute once, not per light
   
   for light in lights:
       # ... rest of code using view_dir
   ```

2. **Pre-check reflection color** (in material or during setup):
   ```python
   # In Material.__init__ or in render_scene setup:
   material.has_reflection = np.any(material.reflection_color > 0)
   
   # In trace_ray, replace line 283:
   if material.has_reflection and depth < scene_settings.max_recursions:
   ```

3. **Pre-compute material arrays** (Priority 1 already covers this)

---

## Priority 6: Optimize `find_closest_intersection` Loop (HARD, MEDIUM IMPACT)

### Problem
- Line 197: `if hit and t < closest_t` checked 87.9M times
- Loop iterates through all surfaces for every ray
- No early termination or spatial optimization

**Estimated savings: 5-10% speedup (if implemented well)**

### Implementation Details

1. **Early termination optimization**:
   ```python
   # If we hit a surface very close (t < some threshold), return immediately
   # Only useful if surfaces are sorted or if t values are typically small
   # This is risky - might skip closer surfaces, so be careful
   ```

2. **Separate surface lists by type**:
   ```python
   # In render_scene setup:
   spheres = [s for s in surfaces if hasattr(s, 'radius')]
   planes = [s for s in surfaces if hasattr(s, 'normal')]
   cubes = [s for s in surfaces if hasattr(s, 'scale')]
   
   # In find_closest_intersection:
   for sphere in spheres:
       hit, t, point, normal = intersect_sphere(...)
       # ... update closest
   for plane in planes:
       hit, t, point, normal = intersect_plane(...)
       # ... update closest
   # etc.
   ```
   **Benefit**: Eliminates `hasattr()` checks (line 190, 192, 194) - saves ~1.8% of function time

3. **Batch intersection testing** (ADVANCED - vectorize multiple rays):
   - Process multiple rays at once using numpy broadcasting
   - Complex but could yield 2-5x speedup
   - Requires significant refactoring

**Recommendation: Start with #2 (moderate refactoring, good payoff)**

---

## Implementation Order (Recommended)

### Phase 1: Quick Wins (1-2 hours, ~20-30% speedup)
1. ✅ Priority 1: Eliminate redundant `np.array()` conversions
2. ✅ Priority 2: Optimize `normalize()` function (Option A)
3. ✅ Priority 5: Cache view_dir and reflection checks

### Phase 2: Medium Effort (2-3 hours, +10-15% speedup)
4. ✅ Priority 3: Optimize `cast_shadow_rays` (#4 - combine distance/norm)
5. ✅ Priority 4: Optimize `intersect_sphere` normal calculation

### Phase 3: Advanced (3-5 hours, +5-10% speedup)
6. ✅ Priority 3: Vectorize shadow ray calculations (#1, #3)
7. ✅ Priority 6: Separate surface lists by type

### Phase 4: Experimental (variable time)
8. ✅ Priority 4: Verify ray normalization assumption
9. ✅ Priority 6: Batch processing / spatial structures

---

## Testing Strategy

After each optimization:
1. Run with small image (200x200) to verify correctness
2. Compare output images (should be identical or very close)
3. Measure speedup with profiling
4. Run full-size test (500x500 or 1000x1000) to confirm

## Expected Total Speedup

**Conservative estimate: 1.5-2.5x faster**
- Phase 1: 1.2-1.3x
- Phase 2: 1.1-1.15x  
- Phase 3: 1.05-1.1x
- **Total: ~1.4-1.65x**

**Optimistic estimate: 2-4x faster**
- If vectorization works well: additional 1.2-1.5x
- **Total: ~2.4-2.6x**

---

## Code Quality Notes

- Maintain readability - add comments for optimizations
- Keep original code structure if possible
- Consider making optimizations optional (flags) for debugging
- Profile after each change to verify improvements

