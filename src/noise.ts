import * as THREE from 'three/webgpu';
import { ImprovedNoise } from 'three/addons/math/ImprovedNoise.js';
import type { CloudVolumeOptions } from './types';

/**
 * Generates the main 3D volume texture with multi-octave FBM,
 * optional billow/turbulence shaping, and isotropic tendrils/falloff.
 */
export function generateCloudVolumeData( {
	size = 128,
	scale = 0.055,
	octaves = 4,
	lacunarity = 2.05,
	gain = 0.5,
	turbulence = 0.45,
	falloffPower = 1.65,
	tendrils = 0.72,
	seed = 13.7
}: CloudVolumeOptions = {} ): Uint8Array {

	const data = new Uint8Array( size * size * size );
	const perlin = new ImprovedNoise();
	const vector = new THREE.Vector3();

	const invSize = 1 / size;

	const seedX = seed * 17.13;
	const seedY = seed * 31.71;
	const seedZ = seed * 47.37;

	let i = 0;

	for ( let z = 0; z < size; z ++ ) {

		const nz = z * invSize - 0.5;

		for ( let y = 0; y < size; y ++ ) {

			const ny = y * invSize - 0.5;

			for ( let x = 0; x < size; x ++ ) {

				const nx = x * invSize - 0.5;

				// Soft margin at box edges so tendrils never clip flat
				const boxEdge = Math.max( Math.abs( nx ), Math.abs( ny ), Math.abs( nz ) );
				if ( boxEdge > 0.485 ) {

					data[ i ++ ] = 0;
					continue;

				}

				const edgeFade = Math.min( 1.0, ( 0.485 - boxEdge ) / 0.075 );

				// Isotropic low-frequency branch/tendril warp (no axis stretching!)
				const branchFreq = 2.8;
				const b1 = perlin.noise( nx * branchFreq + seedX, ny * branchFreq + seedY, nz * branchFreq + seedZ );
				const b2 = perlin.noise( nx * branchFreq * 1.9 + seedX + 11.4, ny * branchFreq * 1.9 + seedY + 5.7, nz * branchFreq * 1.9 + seedZ );
				const branchMod = ( b1 * 0.68 + b2 * 0.32 ) * tendrils;

				// Strictly isotropic spherical distance sculpted into organic arms by branchMod
				vector.set( nx, ny, nz );
				const dist = vector.length() - branchMod * 0.24;
				const d = Math.max( 0.0, 1.0 - Math.max( 0.0, dist ) * 2.1 );

				if ( d <= 0.0 ) {

					data[ i ++ ] = 0;
					continue;

				}

				// Isotropic multi-octave FBM + billow turbulence
				let amp = 1.0;
				let freq = scale;
				let sum = 0.0;
				let norm = 0.0;

				for ( let o = 0; o < octaves; o ++ ) {

					const n = perlin.noise(
						x * freq + seedX,
						y * freq + seedY,
						z * freq + seedZ
					);

					// Billowy smoke shape: 1 - |n| creates plump ridges, standard n is soft
					const billow = 1.0 - Math.abs( n ) * 1.65;
					const blended = n * ( 1.0 - turbulence ) + billow * turbulence;

					sum += blended * amp;
					norm += amp;

					freq *= lacunarity;
					amp *= gain;

				}

				const normalized = ( sum / norm ) * 0.5 + 0.5;
				const shaped = Math.max( 0, Math.min( 1, normalized ) ) * Math.pow( d, falloffPower ) * edgeFade;
				data[ i ++ ] = Math.floor( shaped * 255 );

			}

		}

	}

	return data;

}

export function createCloudVolumeTexture( options: CloudVolumeOptions = {} ): THREE.Data3DTexture {

	const size = options.size ?? 128;
	const data = generateCloudVolumeData( options );

	const texture = new THREE.Data3DTexture( data, size, size, size );
	texture.format = THREE.RedFormat;
	texture.minFilter = THREE.LinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.wrapS = THREE.ClampToEdgeWrapping;
	texture.wrapT = THREE.ClampToEdgeWrapping;
	texture.wrapR = THREE.ClampToEdgeWrapping;
	texture.unpackAlignment = 1;
	texture.needsUpdate = true;

	return texture;

}

/**
 * Generates a seamlessly tileable RGBA 3D texture:
 * - RGB: 3D warp vector for domain-warped swirling smoke
 * - A:   high-frequency detail noise for edge erosion
 * Packing all 4 channels into one RGBA texture cuts shader fetches by 4x.
 */
export function createDetailVolumeTexture( size = 64 ): THREE.Data3DTexture {

	const data = new Uint8Array( size * size * size * 4 );
	const perlin = new ImprovedNoise();

	const samplePeriodic = (
		nx: number,
		ny: number,
		nz: number,
		freq: number,
		ox: number,
		oy: number,
		oz: number
	): number => {

		const fx = nx * freq + ox;
		const fy = ny * freq + oy;
		const fz = nz * freq + oz;

		const p = freq;
		const wx = nx;
		const wy = ny;
		const wz = nz;

		const n000 = perlin.noise( fx, fy, fz );
		const n100 = perlin.noise( fx - p, fy, fz );
		const n010 = perlin.noise( fx, fy - p, fz );
		const n110 = perlin.noise( fx - p, fy - p, fz );
		const n001 = perlin.noise( fx, fy, fz - p );
		const n101 = perlin.noise( fx - p, fy, fz - p );
		const n011 = perlin.noise( fx, fy - p, fz - p );
		const n111 = perlin.noise( fx - p, fy - p, fz - p );

		const c00 = n000 * ( 1 - wx ) + n100 * wx;
		const c10 = n010 * ( 1 - wx ) + n110 * wx;
		const c01 = n001 * ( 1 - wx ) + n101 * wx;
		const c11 = n011 * ( 1 - wx ) + n111 * wx;

		const c0 = c00 * ( 1 - wy ) + c10 * wy;
		const c1 = c01 * ( 1 - wy ) + c11 * wy;

		return c0 * ( 1 - wz ) + c1 * wz;

	};

	const toByte = ( v: number ): number => Math.floor( Math.max( 0, Math.min( 1, v ) ) * 255 );

	let i = 0;

	for ( let z = 0; z < size; z ++ ) {

		const nz = z / size;

		for ( let y = 0; y < size; y ++ ) {

			const ny = y / size;

			for ( let x = 0; x < size; x ++ ) {

				const nx = x / size;

				// RGB: 3 independent low/mid-freq periodic channels for 3D domain warp
				const wx = samplePeriodic( nx, ny, nz, 4.0, 0.0, 0.0, 0.0 ) * 0.85 + 0.5;
				const wy = samplePeriodic( nx, ny, nz, 4.0, 17.3, 5.1, 11.7 ) * 0.85 + 0.5;
				const wz = samplePeriodic( nx, ny, nz, 4.0, 31.9, 23.4, 7.2 ) * 0.85 + 0.5;

				// A: multi-octave wispy detail
				const o1 = samplePeriodic( nx, ny, nz, 4.0, 3.7, 13.1, 19.3 ) * 0.62;
				const o2 = samplePeriodic( nx, ny, nz, 8.0, 11.2, 29.5, 5.8 ) * 0.28;
				const o3 = samplePeriodic( nx, ny, nz, 16.0, 2.3, 7.9, 17.1 ) * 0.10;
				const detail = ( o1 + o2 + o3 ) * 0.8 + 0.5;

				data[ i ++ ] = toByte( wx );
				data[ i ++ ] = toByte( wy );
				data[ i ++ ] = toByte( wz );
				data[ i ++ ] = toByte( detail );

			}

		}

	}

	const texture = new THREE.Data3DTexture( data, size, size, size );
	texture.format = THREE.RGBAFormat;
	texture.minFilter = THREE.LinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.wrapS = THREE.RepeatWrapping;
	texture.wrapT = THREE.RepeatWrapping;
	texture.wrapR = THREE.RepeatWrapping;
	texture.unpackAlignment = 1;
	texture.needsUpdate = true;

	return texture;

}
