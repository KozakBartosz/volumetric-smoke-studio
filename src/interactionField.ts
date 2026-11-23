import * as THREE from 'three/webgpu';

export interface StampParams {
	radius: number;
	push: number;
	drag: number;
	swirl: number;
	disperse: number;
}

/**
 * Lightweight 32x32x32 3D fluid wake field painted by mouse motion.
 * - RGB: 3D advection displacement vector (encoded [-1, 1] -> [0, 255], 128 = 0)
 * - A:   center dispersion / thinning factor ([0, 1] -> [0, 255])
 *
 * Fits in ~128 KB (stays hot in GPU L2 cache) and gives O(1) shader cost
 * regardless of how many overlapping strokes are painted.
 */
export class SmokeInteractionField {

	readonly size: number;
	readonly texture: THREE.Data3DTexture;

	private readonly floatData: Float32Array;
	private readonly byteData: Uint8Array;
	private readonly voxelCount: number;
	private active = false;
	private maxEnergy = 0;

	constructor( size = 32 ) {

		this.size = size;
		this.voxelCount = size * size * size;
		this.floatData = new Float32Array( this.voxelCount * 4 );
		this.byteData = new Uint8Array( this.voxelCount * 4 );

		// Initialize neutral displacement (128, 128, 128) and zero dispersion (0)
		for ( let i = 0; i < this.voxelCount; i ++ ) {

			const o = i * 4;
			this.byteData[ o ] = 128;
			this.byteData[ o + 1 ] = 128;
			this.byteData[ o + 2 ] = 128;
			this.byteData[ o + 3 ] = 0;

		}

		this.texture = new THREE.Data3DTexture( this.byteData, size, size, size );
		this.texture.format = THREE.RGBAFormat;
		this.texture.minFilter = THREE.LinearFilter;
		this.texture.magFilter = THREE.LinearFilter;
		this.texture.wrapS = THREE.ClampToEdgeWrapping;
		this.texture.wrapT = THREE.ClampToEdgeWrapping;
		this.texture.wrapR = THREE.ClampToEdgeWrapping;
		this.texture.unpackAlignment = 1;
		this.texture.needsUpdate = true;

	}

	get isActive(): boolean {

		return this.active;

	}

	get activityLevel(): number {

		return this.maxEnergy;

	}

	/**
	 * Paints a continuous wind/wake stroke between two local-space points in [-0.5, 0.5]^3.
	 */
	paintStroke( from: THREE.Vector3, to: THREE.Vector3, params: StampParams, intensity = 1.0 ): void {

		if ( intensity <= 0.01 ) return;

		const dx = to.x - from.x;
		const dy = to.y - from.y;
		const dz = to.z - from.z;
		const segLen = Math.hypot( dx, dy, dz );
		if ( segLen < 0.001 ) return;

		// Normalized swipe direction + clamped speed factor
		const invLen = 1 / segLen;
		const vx = dx * invLen;
		const vy = dy * invLen;
		const vz = dz * invLen;
		const speedScale = Math.min( 1.0, 0.35 + segLen * 8.0 ) * intensity;

		// Subdivide fast swipes so the painted trail is always seamless
		const stepDist = Math.max( params.radius * 0.42, 0.012 );
		const steps = Math.min( 16, Math.max( 1, Math.ceil( segLen / stepDist ) ) );
		const strokeWeight = Math.min( 1.0, 1.35 / Math.sqrt( steps ) );

		for ( let s = 1; s <= steps; s ++ ) {

			const t = s / steps;
			const cx = from.x + dx * t;
			const cy = from.y + dy * t;
			const cz = from.z + dz * t;

			this.stampPoint( cx, cy, cz, vx, vy, vz, speedScale * strokeWeight, params );

		}

	}

	private stampPoint(
		cx: number,
		cy: number,
		cz: number,
		vx: number,
		vy: number,
		vz: number,
		strength: number,
		params: StampParams
	): void {

		const N = this.size;
		const r = Math.max( params.radius, 0.015 );
		const influenceR = r * 2.35;
		const invRSq = 1 / ( r * r );
		const invCenterRSq = 1 / ( r * r * 0.52 );

		// Tight voxel AABB around the brush point
		const minX = Math.max( 0, Math.floor( ( cx + 0.5 - influenceR ) * N ) );
		const maxX = Math.min( N - 1, Math.ceil( ( cx + 0.5 + influenceR ) * N ) );
		const minY = Math.max( 0, Math.floor( ( cy + 0.5 - influenceR ) * N ) );
		const maxY = Math.min( N - 1, Math.ceil( ( cy + 0.5 + influenceR ) * N ) );
		const minZ = Math.max( 0, Math.floor( ( cz + 0.5 - influenceR ) * N ) );
		const maxZ = Math.min( N - 1, Math.ceil( ( cz + 0.5 + influenceR ) * N ) );

		if ( minX > maxX || minY > maxY || minZ > maxZ ) return;

		const f = this.floatData;
		const invN = 1 / N;
		const pushGain = params.push * strength * 1.45;
		const dragGain = params.drag * strength * 1.15;
		const swirlGain = params.swirl * strength * 3.2;
		const dispGain = params.disperse * strength * 0.9;

		for ( let z = minZ; z <= maxZ; z ++ ) {

			const pz = ( z + 0.5 ) * invN - 0.5;
			const oz = pz - cz;
			const oz2 = oz * oz;

			for ( let y = minY; y <= maxY; y ++ ) {

				const py = ( y + 0.5 ) * invN - 0.5;
				const oy = py - cy;
				const oy2 = oy * oy;

				const rowOffset = ( z * N + y ) * N;

				for ( let x = minX; x <= maxX; x ++ ) {

					const px = ( x + 0.5 ) * invN - 0.5;
					const ox = px - cx;
					const distSq = ox * ox + oy2 + oz2;

					if ( distSq > influenceR * influenceR ) continue;

					const w = Math.exp( - distSq * invRSq );
					const dist = Math.sqrt( distSq );

					// Smooth radial outward push (peaks just outside center, zero at exact center)
					const radFactor = ( dist / ( dist * dist + r * r * 0.35 ) ) * r;
					const radScale = ( dist > 1e-5 ? ( radFactor / dist ) : 0 ) * pushGain * w;

					// Twin counter-rotating Karman eddies: cross(vel, offset), boosted with radial ring falloff
					const ringBoost = 1.0 + ( dist / ( r + 1e-4 ) ) * 0.85;
					const eddyScale = swirlGain * 1.65 * w * ringBoost;
					const sx = ( vy * oz - vz * oy ) * eddyScale;
					const sy = ( vz * ox - vx * oz ) * eddyScale;
					const sz = ( vx * oy - vy * ox ) * eddyScale;

					// Organic micro-curl perturbation around the brush so the wake twists unpredictably
					const turbPhase = px * 26.0 + py * 21.0 + pz * 29.0;
					const turbScale = swirlGain * 0.14 * w;
					const tx = Math.sin( turbPhase + oy * 35.0 ) * turbScale;
					const ty = Math.cos( turbPhase + oz * 35.0 ) * turbScale;
					const tz = Math.sin( turbPhase * 0.7 - ox * 35.0 ) * turbScale;

					// Combined displacement delta
					const addX = ox * radScale + vx * dragGain * w + sx + tx;
					const addY = oy * radScale + vy * dragGain * w + sy + ty;
					const addZ = oz * radScale + vz * dragGain * w + sz + tz;

					// Center thinning/dispersion (tighter Gaussian in the middle of the stroke)
					const wCenter = Math.exp( - distSq * invCenterRSq ) * dispGain;

					const idx = ( rowOffset + x ) * 4;

					// Soft-saturating accumulation so overlapping strokes stay smooth & bounded [-1, 1]
					const nx = f[ idx ] + addX * ( 1 - Math.abs( f[ idx ] ) * 0.55 );
					const ny = f[ idx + 1 ] + addY * ( 1 - Math.abs( f[ idx + 1 ] ) * 0.55 );
					const nz = f[ idx + 2 ] + addZ * ( 1 - Math.abs( f[ idx + 2 ] ) * 0.55 );
					const na = f[ idx + 3 ] + wCenter * ( 1 - f[ idx + 3 ] );

					f[ idx ] = Math.max( - 1, Math.min( 1, nx ) );
					f[ idx + 1 ] = Math.max( - 1, Math.min( 1, ny ) );
					f[ idx + 2 ] = Math.max( - 1, Math.min( 1, nz ) );
					f[ idx + 3 ] = Math.max( 0, Math.min( 1, na ) );

				}

			}

		}

		this.active = true;
		this.maxEnergy = 1.0;

	}

	/**
	 * Stirs and parts smoke using the EXACT oriented bounding box (OBB) shape of a 3D cube:
	 * - Exact OBB signed distance & face/edge/corner normals
	 * - Rigid-body velocity per voxel: v_linear + cross(omega, r) (strongest at the 8 rotating corners!)
	 * Uses the exact same 32^3 field texture -> 0 extra GPU cost, < 0.03 ms CPU.
	 */
	stampOrientedBox(
		pos: THREE.Vector3,
		invBoxMatrix3: THREE.Matrix3,
		boxMatrix3: THREE.Matrix3,
		halfExtent: number,
		wakeSpread: number,
		vel: THREE.Vector3,
		omega: THREE.Vector3,
		strength: number,
		dt: number
	): void {

		if ( strength <= 0.001 ) return;

		const N = this.size;
		// wakeSpread is centered at 0:
		// - negative shrinks the disturbance box inward (ultra-tight hug / corners only)
		// - 0 sits tightly on the cube surface
		// - positive expands the wake aura outward
		const effectiveHalf = Math.max( 0.008, halfExtent + Math.min( 0, wakeSpread ) );
		const shell = Math.max( 0.014, 0.018 + Math.max( 0, wakeSpread ) );
		const boundR = effectiveHalf * 1.732 + shell * 1.8;
		const invShellSq = 1 / ( shell * shell );

		const minX = Math.max( 0, Math.floor( ( pos.x + 0.5 - boundR ) * N ) );
		const maxX = Math.min( N - 1, Math.ceil( ( pos.x + 0.5 + boundR ) * N ) );
		const minY = Math.max( 0, Math.floor( ( pos.y + 0.5 - boundR ) * N ) );
		const maxY = Math.min( N - 1, Math.ceil( ( pos.y + 0.5 + boundR ) * N ) );
		const minZ = Math.max( 0, Math.floor( ( pos.z + 0.5 - boundR ) * N ) );
		const maxZ = Math.min( N - 1, Math.ceil( ( pos.z + 0.5 + boundR ) * N ) );

		if ( minX > maxX || minY > maxY || minZ > maxZ ) return;

		const f = this.floatData;
		const invN = 1 / N;
		const rate = Math.min( 1.0, dt * 9.0 ) * strength;

		const im = invBoxMatrix3.elements;
		const bm = boxMatrix3.elements;

		const pushGain = 0.52 * rate;
		const velGain = 1.35 * rate;
		const dispGain = 0.34 * rate;
		const maxDist = shell * 1.85;

		for ( let z = minZ; z <= maxZ; z ++ ) {

			const pz = ( z + 0.5 ) * invN - 0.5;
			const oz = pz - pos.z;

			for ( let y = minY; y <= maxY; y ++ ) {

				const py = ( y + 0.5 ) * invN - 0.5;
				const oy = py - pos.y;
				const rowOffset = ( z * N + y ) * N;

				for ( let x = minX; x <= maxX; x ++ ) {

					const px = ( x + 0.5 ) * invN - 0.5;
					const ox = px - pos.x;

					// Transform offset into the cube's local oriented frame (column-major Matrix3)
					const lx = im[ 0 ] * ox + im[ 3 ] * oy + im[ 6 ] * oz;
					const ly = im[ 1 ] * ox + im[ 4 ] * oy + im[ 7 ] * oz;
					const lz = im[ 2 ] * ox + im[ 5 ] * oy + im[ 8 ] * oz;

					// Exact OBB SDF: distance from effective box faces, edges & 8 corners
					const ax = Math.abs( lx ) - effectiveHalf;
					const ay = Math.abs( ly ) - effectiveHalf;
					const az = Math.abs( lz ) - effectiveHalf;

					const qx = ax > 0 ? ax : 0;
					const qy = ay > 0 ? ay : 0;
					const qz = az > 0 ? az : 0;
					const outDist = Math.hypot( qx, qy, qz );

					if ( outDist > maxDist ) continue;

					// Smooth boundary falloff hugging the cube's faces & corners
					const w = Math.exp( - ( outDist * outDist ) * invShellSq );

					// Exact outward surface normal in cube-local space
					let nlx = 0;
					let nly = 0;
					let nlz = 0;

					if ( outDist > 1e-5 ) {

						const invOut = 1 / outDist;
						nlx = ( lx >= 0 ? qx : - qx ) * invOut;
						nly = ( ly >= 0 ? qy : - qy ) * invOut;
						nlz = ( lz >= 0 ? qz : - qz ) * invOut;

					} else {

						// Inside cube: push out along nearest face normal
						if ( ax >= ay && ax >= az ) nlx = lx >= 0 ? 1 : - 1;
						else if ( ay >= az ) nly = ly >= 0 ? 1 : - 1;
						else nlz = lz >= 0 ? 1 : - 1;

					}

					// Rotate normal back into cloud local space
					const nx = bm[ 0 ] * nlx + bm[ 3 ] * nly + bm[ 6 ] * nlz;
					const ny = bm[ 1 ] * nlx + bm[ 4 ] * nly + bm[ 7 ] * nlz;
					const nz = bm[ 2 ] * nlx + bm[ 5 ] * nly + bm[ 8 ] * nlz;

					// True rigid-body velocity at this exact point: v_linear + cross(omega, offset)
					// Naturally peaks at the 8 rotating corners!
					const vx = vel.x + ( omega.y * oz - omega.z * oy );
					const vy = vel.y + ( omega.z * ox - omega.x * oz );
					const vz = vel.z + ( omega.x * oy - omega.y * ox );

					// Extra corner boost where point is near a corner (high |lx|+|ly|+|lz|)
					const cornerFactor = 0.75 + 0.55 * Math.min( 1.0, ( Math.abs( lx ) + Math.abs( ly ) + Math.abs( lz ) ) / ( halfExtent * 2.6 ) );

					const addX = ( nx * pushGain + vx * velGain * cornerFactor ) * w;
					const addY = ( ny * pushGain + vy * velGain * cornerFactor ) * w;
					const addZ = ( nz * pushGain + vz * velGain * cornerFactor ) * w;

					const idx = ( rowOffset + x ) * 4;

					const fx = f[ idx ] + addX * ( 1 - Math.abs( f[ idx ] ) * 0.65 );
					const fy = f[ idx + 1 ] + addY * ( 1 - Math.abs( f[ idx + 1 ] ) * 0.65 );
					const fz = f[ idx + 2 ] + addZ * ( 1 - Math.abs( f[ idx + 2 ] ) * 0.65 );
					const fa = f[ idx + 3 ] + dispGain * w * ( 1 - f[ idx + 3 ] );

					f[ idx ] = Math.max( - 1, Math.min( 1, fx ) );
					f[ idx + 1 ] = Math.max( - 1, Math.min( 1, fy ) );
					f[ idx + 2 ] = Math.max( - 1, Math.min( 1, fz ) );
					f[ idx + 3 ] = Math.max( 0, Math.min( 0.65, fa ) );

				}

			}

		}

		this.active = true;
		this.maxEnergy = Math.max( this.maxEnergy, 0.5 );

	}

	/**
	 * Smoothly relaxes the painted 3D field back to rest and uploads to GPU only when active.
	 */
	update( dt: number, fadeSpeed: number ): void {

		if ( ! this.active ) return;

		const decay = Math.exp( - dt * fadeSpeed );
		const dispDecay = Math.exp( - dt * fadeSpeed * 1.15 );
		const f = this.floatData;
		const b = this.byteData;
		const len = this.voxelCount * 4;

		let peak = 0;

		for ( let i = 0; i < len; i += 4 ) {

			const fx = f[ i ] * decay;
			const fy = f[ i + 1 ] * decay;
			const fz = f[ i + 2 ] * decay;
			const fa = f[ i + 3 ] * dispDecay;

			f[ i ] = fx;
			f[ i + 1 ] = fy;
			f[ i + 2 ] = fz;
			f[ i + 3 ] = fa;

			const m = Math.max( Math.abs( fx ), Math.abs( fy ), Math.abs( fz ), fa );
			if ( m > peak ) peak = m;

			b[ i ] = Math.round( ( fx * 0.5 + 0.5 ) * 255 );
			b[ i + 1 ] = Math.round( ( fy * 0.5 + 0.5 ) * 255 );
			b[ i + 2 ] = Math.round( ( fz * 0.5 + 0.5 ) * 255 );
			b[ i + 3 ] = Math.round( fa * 255 );

		}

		this.maxEnergy = peak;

		if ( peak < 0.003 ) {

			f.fill( 0 );
			for ( let i = 0; i < len; i += 4 ) {

				b[ i ] = 128;
				b[ i + 1 ] = 128;
				b[ i + 2 ] = 128;
				b[ i + 3 ] = 0;

			}
			this.active = false;
			this.maxEnergy = 0;

		}

		this.texture.needsUpdate = true;

	}

	clear(): void {

		this.floatData.fill( 0 );
		const len = this.voxelCount * 4;
		for ( let i = 0; i < len; i += 4 ) {

			this.byteData[ i ] = 128;
			this.byteData[ i + 1 ] = 128;
			this.byteData[ i + 2 ] = 128;
			this.byteData[ i + 3 ] = 0;

		}
		this.active = false;
		this.maxEnergy = 0;
		this.texture.needsUpdate = true;

	}

}
