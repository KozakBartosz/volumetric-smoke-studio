import * as THREE from 'three/webgpu';
import {
	Break,
	Fn,
	If,
	Loop,
	cameraFar,
	cameraNear,
	cameraPosition,
	clamp,
	dot,
	exp,
	float,
	fract,
	length,
	max,
	min,
	mix,
	modelWorldMatrixInverse,
	perspectiveDepthToViewZ,
	positionGeometry,
	positionView,
	pow,
	screenCoordinate,
	smoothstep,
	texture3D,
	uniform,
	varying,
	vec2,
	vec3,
	vec4,
	viewportDepthTexture
} from 'three/tsl';
import type { CloudState } from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const hitBox = /*@__PURE__*/ Fn( ( { orig, dir }: { orig: any; dir: any } ) => {

	const boxMin = vec3( - 0.5 );
	const boxMax = vec3( 0.5 );

	const invDir = dir.reciprocal();
	const tminTmp = boxMin.sub( orig ).mul( invDir );
	const tmaxTmp = boxMax.sub( orig ).mul( invDir );

	const tmin = min( tminTmp, tmaxTmp );
	const tmax = max( tminTmp, tmaxTmp );

	const t0 = max( tmin.x, max( tmin.y, tmin.z ) );
	const t1 = min( tmax.x, min( tmax.y, tmax.z ) );

	return vec2( t0, t1 );

}, { orig: 'vec3', dir: 'vec3', return: 'vec2' } );

export function createVolumetricCloudMaterial(
	baseTexture: THREE.Data3DTexture,
	detailTexture: THREE.Data3DTexture,
	interactionTexture: THREE.Data3DTexture,
	initialParams: CloudState
) {

	const uniforms = {
		// Colors
		shadowColor: uniform( new THREE.Color( initialParams.shadowColor ) ),
		midColor: uniform( new THREE.Color( initialParams.midColor ) ),
		highlightColor: uniform( new THREE.Color( initialParams.highlightColor ) ),
		rimColor: uniform( new THREE.Color( initialParams.rimColor ) ),
		coreGlowColor: uniform( new THREE.Color( initialParams.coreGlowColor ) ),
		ambientColor: uniform( new THREE.Color( initialParams.ambientColor ) ),

		// Lighting & Shadows
		lightColor: uniform( new THREE.Color( initialParams.lightColor ) ),
		lightDir: uniform( new THREE.Vector3( 0.5, 0.7, 0.5 ).normalize() ),
		lightIntensity: uniform( initialParams.lightIntensity ),
		ambientIntensity: uniform( initialParams.ambientIntensity ),
		shadowAbsorption: uniform( initialParams.shadowAbsorption ),
		shadowDistance: uniform( initialParams.shadowDistance ),
		shadowContrast: uniform( initialParams.shadowContrast ),
		reliefStrength: uniform( initialParams.reliefStrength ),
		powderStrength: uniform( initialParams.powderStrength ),
		rimIntensity: uniform( initialParams.rimIntensity ),
		anisotropy: uniform( initialParams.anisotropy ),
		coreGlowIntensity: uniform( initialParams.coreGlowIntensity ),

		// Density & Shape
		threshold: uniform( initialParams.threshold ),
		range: uniform( initialParams.range ),
		opacity: uniform( initialParams.opacity ),
		absorption: uniform( initialParams.absorption ),
		steps: uniform( initialParams.steps ),
		jitter: uniform( initialParams.jitter ),
		exposure: uniform( initialParams.exposure ),
		contrast: uniform( initialParams.contrast ),

		// Detail & Motion
		detailScale: uniform( initialParams.detailScale ),
		detailStrength: uniform( initialParams.detailStrength ),
		warpStrength: uniform( initialParams.warpStrength ),
		warpScale: uniform( initialParams.warpScale ),
		windOffset: uniform( new THREE.Vector3( 0, 0, 0 ) ),

		// Painted 3D Fluid Wake Field (O(1) single texture fetch per active step)
		mouseActive: uniform( 0.0 ),
		mouseDisperse: uniform( initialParams.mouseDisperse ),
		mouseTurbulence: uniform( initialParams.mouseTurbulence ),
		mouseClampToSmoke: uniform( initialParams.mouseClampToSmoke ? 1.0 : 0.0 )
	};

	const baseTexNode = texture3D( baseTexture, null, 0 );
	const detailTexNode = texture3D( detailTexture, null, 0 );
	const interactionTexNode = texture3D( interactionTexture, null, 0 );

	// Henyey-Greenstein phase function (computed once per pixel)
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const hgPhase = Fn( ( [ cosTheta, g ]: [ any, any ] ) => {

		const g2 = g.mul( g );
		const denom = pow( float( 1.0 ).add( g2 ).sub( g.mul( 2.0 ).mul( cosTheta ) ), 1.5 );
		return float( 1.0 ).sub( g2 ).div( max( denom, 0.001 ) ).mul( 0.25 );

	} );

	const volumetricSmokeShader = Fn( () => {

		const vOrigin = varying( vec3( modelWorldMatrixInverse.mul( vec4( cameraPosition, 1.0 ) ) ) );
		const vDirection = varying( positionGeometry.sub( vOrigin ) );
		const rayDir = vDirection.normalize();

		const bounds = vec2( hitBox( { orig: vOrigin, dir: rayDir } ) ).toVar();
		bounds.x.greaterThan( bounds.y ).discard();
		bounds.assign( vec2( max( bounds.x, 0.0 ), bounds.y ) );

		// Depth-aware raymarching: stop the ray at opaque scene geometry
		const sceneDepth = viewportDepthTexture();
		const sceneViewZ = perspectiveDepthToViewZ( sceneDepth, cameraNear, cameraFar );
		const tScene = length( vDirection ).mul( sceneViewZ.div( positionView.z ) );
		const tEnd = min( bounds.y, tScene ).toVar();

		bounds.x.greaterThanEqual( tEnd ).discard();

		const inc = vec3( rayDir.abs().reciprocal() );
		const stepSize = float( min( inc.x, min( inc.y, inc.z ) ) ).div( uniforms.steps );

		// Interleaved Gradient Noise ray jitter eliminates banding at low step counts
		const ign = fract(
			float( 52.9829189 ).mul(
				fract( dot( screenCoordinate.xy, vec2( 0.06711056, 0.00583715 ) ) )
			)
		);
		const startOffset = ign.mul( stepSize ).mul( uniforms.jitter );
		const tStart = bounds.x.add( startOffset );
		const positionRay = vec3( vOrigin.add( tStart.mul( rayDir ) ) ).toVar();

		// Precompute per-pixel constants outside the loop
		const cosTheta = dot( rayDir, uniforms.lightDir );
		const phase = clamp( hgPhase( cosTheta, uniforms.anisotropy ), 0.0, 4.0 );
		const rimBase = uniforms.rimColor.mul( phase ).mul( uniforms.rimIntensity );
		const lightTint = mix( vec3( 1.0 ), uniforms.lightColor, 0.65 );
		const edgeLow = uniforms.threshold.sub( uniforms.range );
		const edgeHigh = uniforms.threshold.add( uniforms.range );
		const emptyCutoff = max( edgeLow.sub( 0.14 ), 0.01 );

		const dStep1 = uniforms.lightDir.mul( uniforms.shadowDistance.mul( 0.08 ) );
		const dStep2 = uniforms.lightDir.mul( uniforms.shadowDistance.mul( 0.28 ) );
		const dStep3 = uniforms.lightDir.mul( uniforms.shadowDistance.mul( 0.65 ) );

		const finalColor = vec4( 0.0 ).toVar();

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		Loop( { type: 'float', start: tStart, end: tEnd, update: stepSize }, ( { i: tCur }: { i: any } ) => {

			// Clamp last step length right up to the opaque surface for silky-smooth intersections
			const localStep = min( stepSize, max( tEnd.sub( tCur ), 0.0 ) );

			const baseUvw = positionRay.add( 0.5 );
			const advectedUvw = vec3( baseUvw ).toVar();
			const centerDisperse = float( 0.0 ).toVar();
			const wakeEnergy = float( 0.0 ).toVar();
			const origMask = float( 1.0 ).toVar();

			// O(1) painted 3D wind/wake field lookup (skipped completely when idle!)
			If( uniforms.mouseActive.greaterThan( 0.001 ), () => {

				// Optional mask prevents pulling ghost smoke into empty air outside the cloud
				If( uniforms.mouseClampToSmoke.greaterThan( 0.5 ), () => {

					const origVal = float( baseTexNode.sample( baseUvw ).r );
					origMask.assign( smoothstep( edgeLow.sub( 0.06 ), edgeLow.add( 0.05 ), origVal ) );

				} );

				If( origMask.greaterThan( 0.001 ), () => {

					const fieldSample = interactionTexNode.sample( baseUvw );
					const rawDisp = fieldSample.rgb.sub( 0.5 ).mul( 2.0 );
					const disp = rawDisp.mul( 0.36 ).mul( origMask );
					advectedUvw.subAssign( disp );
					centerDisperse.assign( fieldSample.a.mul( uniforms.mouseDisperse ) );
					wakeEnergy.assign( clamp( max( length( rawDisp ), fieldSample.a ), 0.0, 1.0 ).mul( origMask ) );

				} );

			} );

			// Fast coarse empty-space check on advected coordinates (1 fetch in empty air)
			const coarseVal = float( baseTexNode.sample( advectedUvw ).r );

			If( coarseVal.greaterThan( emptyCutoff ).and( origMask.greaterThan( 0.001 ) ), () => {

				// Single packed RGBA fetch: RGB = 3D warp vector, A = detail erosion
				// Locally boosts swirling turbulence right around the cursor wake (0 extra fetches!)
				const detailSample = detailTexNode.sample( advectedUvw.mul( uniforms.detailScale ).add( uniforms.windOffset ) );
				const localWarp = uniforms.warpStrength.add( wakeEnergy.mul( uniforms.mouseTurbulence ) );
				const warpVec = detailSample.rgb.sub( 0.5 ).mul( localWarp );
				const warpedUvw = advectedUvw.add( warpVec );

				const baseVal = float( baseTexNode.sample( warpedUvw ).r );
				const eroded = baseVal
					.sub( detailSample.a.oneMinus().mul( uniforms.detailStrength ).mul( 0.48 ) )
					.sub( centerDisperse.mul( 0.38 ) );
				const density = smoothstep( edgeLow, edgeHigh, eroded ).mul( uniforms.opacity ).mul( origMask ).toVar();

				If( density.greaterThan( 0.002 ), () => {

					// 3-tap exponentially spaced shadow march on base volume (3 fetches total)
					const s1 = smoothstep( edgeLow, edgeHigh, float( baseTexNode.sample( warpedUvw.add( dStep1 ) ).r ) );
					const s2 = smoothstep( edgeLow, edgeHigh, float( baseTexNode.sample( warpedUvw.add( dStep2 ) ).r ) );
					const s3 = smoothstep( edgeLow, edgeHigh, float( baseTexNode.sample( warpedUvw.add( dStep3 ) ).r ) );

					const shadowAccum = s1.mul( 0.18 ).add( s2.mul( 0.34 ) ).add( s3.mul( 0.48 ) )
						.mul( uniforms.opacity )
						.mul( uniforms.shadowDistance )
						.mul( 5.2 );

					const lightTransmittance = exp( shadowAccum.negate().mul( uniforms.shadowAbsorption ) );

					// Free micro-relief directional derivative using tap 1 (0 extra fetches)
					const relief = clamp( eroded.sub( s1 ).mul( uniforms.reliefStrength ).add( 0.5 ), 0.0, 1.0 );

					// Powder effect darkens dense smoke creases
					const powder = mix( float( 1.0 ), float( 1.0 ).sub( exp( density.negate().mul( uniforms.powderStrength ).mul( 3.0 ) ) ), 0.45 );

					const rawLit = lightTransmittance.mul( mix( 0.35, 1.0, relief ) ).mul( powder ).mul( uniforms.lightIntensity );
					const lit = pow( clamp( rawLit, 0.0, 1.0 ), uniforms.shadowContrast );

					// 3-stop volumetric color ramp
					const lowToMid = mix( uniforms.shadowColor, uniforms.midColor, smoothstep( 0.0, 0.52, lit ) );
					const smokeAlbedo = mix( lowToMid, uniforms.highlightColor, smoothstep( 0.42, 1.0, lit ) );

					const directLight = smokeAlbedo.mul( lightTint ).mul( mix( 0.35, 1.35, lit ) );

					// Ambient occlusion inside dense volume
					const ambientAO = exp( shadowAccum.negate().mul( 1.1 ) );
					const ambient = uniforms.ambientColor.mul( uniforms.ambientIntensity ).mul( mix( 0.25, 1.0, ambientAO ) );

					// Forward-scattered backlit rim glow
					const rim = rimBase.mul( lightTransmittance );

					// Optional inner core glow
					const coreMask = smoothstep( 0.25, 0.75, density ).mul( lightTransmittance.oneMinus().mul( 0.75 ).add( 0.25 ) );
					const coreGlow = uniforms.coreGlowColor.mul( coreMask ).mul( uniforms.coreGlowIntensity );

					const sampleRgb = directLight.add( ambient ).add( rim ).add( coreGlow );

					// Step-size-independent Beer-Lambert alpha accumulation (clamped cleanly at opaque geometry)
					const stepAlpha = float( 1.0 ).sub( exp( density.negate().mul( uniforms.absorption ).mul( localStep ).mul( 75.0 ) ) );

					const transmittance = finalColor.a.oneMinus();
					finalColor.rgb.addAssign( transmittance.mul( stepAlpha ).mul( sampleRgb ) );
					finalColor.a.addAssign( transmittance.mul( stepAlpha ) );

					If( finalColor.a.greaterThanEqual( 0.96 ), () => {

						Break();

					} );

				} );

			} );

			positionRay.addAssign( rayDir.mul( stepSize ) );

		} );

		// Un-premultiply for clean NormalBlending + exposure/contrast grading
		const safeAlpha = max( finalColor.a, 0.0001 );
		const unpremul = finalColor.rgb.div( safeAlpha );
		const exposed = unpremul.mul( uniforms.exposure );
		const graded = pow( max( exposed, vec3( 0.0 ) ), vec3( uniforms.contrast ) );

		return vec4( graded, finalColor.a );

	} );

	const material = new THREE.NodeMaterial();
	material.colorNode = volumetricSmokeShader();
	material.side = THREE.BackSide;
	material.transparent = true;
	material.depthWrite = false;
	material.depthTest = false;

	return { material, uniforms, baseTexNode };

}

export type CloudShaderUniforms = ReturnType<typeof createVolumetricCloudMaterial>['uniforms'];
