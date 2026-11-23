import './style.css';
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';

import type { CloudState } from './types';
import { PRESETS } from './presets';
import {
	createCloudVolumeTexture,
	createDetailVolumeTexture,
	generateCloudVolumeData
} from './noise';
import {
	createVolumetricCloudMaterial,
	type CloudShaderUniforms
} from './cloudShader';
import { SmokeInteractionField } from './interactionField';

const DEFAULT_PRESET_NAME = 'Dark Amaranth (Dense Smoke)';

const state: CloudState = {
	preset: DEFAULT_PRESET_NAME,
	showLightHelper: false,
	renderScale: 1.0,
	jitter: 1.0,
	showCube: true,
	cubeX: 0.10,
	cubeY: - 0.02,
	cubeZ: 0.08,
	cubeSize: 0.30,
	cubeColor: '#2a2228',
	cubeRotate: true,
	mouseEnabled: true,
	mouseDepthMode: 'dome',
	mouseDomeStrength: 0.5,
	mouseDepthOffset: 0.0,
	mouseClampToSmoke: true,
	mouseRadius: 0.035,
	mousePush: 0.52,
	mouseDrag: 0.89,
	mouseSwirl: 1.0,
	mouseTurbulence: 0.15,
	mouseDisperse: 0.8,
	mouseDelay: 0.05,
	mouseRecovery: 0.52,
	...PRESETS[ DEFAULT_PRESET_NAME ]
};

let renderer: THREE.WebGPURenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let cloudMesh: THREE.Mesh;
let skyCanvas: HTMLCanvasElement;
let skyCtx: CanvasRenderingContext2D;
let skyTexture: THREE.CanvasTexture;
let cloudTexture: THREE.Data3DTexture;
let detailTexture: THREE.Data3DTexture;
let shaderUniforms: CloudShaderUniforms | undefined;
let lightHelperGroup: THREE.Mesh | undefined;
let lightHelperLine: THREE.Line | undefined;
let testCube: THREE.Mesh<THREE.BoxGeometry, THREE.MeshStandardNodeMaterial> | undefined;
let sceneDirLight: THREE.DirectionalLight | undefined;
let sceneAmbLight: THREE.AmbientLight | undefined;
let fpsBadge: HTMLElement | null = null;
let fpsFrames = 0;
let fpsLastTime = performance.now();

const clock = new THREE.Clock();
const worldLightDir = new THREE.Vector3();
const localLightDir = new THREE.Vector3();
const invQuat = new THREE.Quaternion();

// 32x32x32 3D painted fluid wake field + delayed wind gust queue
const interactionField = new SmokeInteractionField( 32 );
interface DelayedWindSample {
	from: THREE.Vector3;
	to: THREE.Vector3;
	camDir: THREE.Vector3;
	mode: CloudState['mouseDepthMode'];
	readyAt: number;
}
const pendingWindQueue: DelayedWindSample[] = [];
let isPointerDown = false;
let hasLastHit = false;
const lastLocalHit = new THREE.Vector3();
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const hitPlane = new THREE.Plane();
const planeNormal = new THREE.Vector3();
const localCamDir = new THREE.Vector3();
const worldHit = new THREE.Vector3();
const localHit = new THREE.Vector3();
const tunnelFrom = new THREE.Vector3();
const tunnelTo = new THREE.Vector3();

function updateSkyGradient(): void {

	const grad = skyCtx.createLinearGradient( 0, 0, 0, skyCanvas.height );
	grad.addColorStop( 0.0, state.bgTop );
	grad.addColorStop( 0.5, state.bgMid );
	grad.addColorStop( 1.0, state.bgBottom );
	skyCtx.fillStyle = grad;
	skyCtx.fillRect( 0, 0, skyCanvas.width, skyCanvas.height );
	skyTexture.needsUpdate = true;

}

function updateLightDirection(): void {

	const az = THREE.MathUtils.degToRad( state.lightAzimuth );
	const el = THREE.MathUtils.degToRad( state.lightElevation );

	const cosEl = Math.cos( el );
	worldLightDir.set(
		Math.sin( az ) * cosEl,
		Math.sin( el ),
		Math.cos( az ) * cosEl
	).normalize();

	if ( lightHelperGroup && lightHelperLine ) {

		lightHelperGroup.visible = state.showLightHelper;
		const pos = worldLightDir.clone().multiplyScalar( 1.15 );
		lightHelperGroup.position.copy( pos );

		const linePos = lightHelperLine.geometry.attributes.position as THREE.BufferAttribute;
		linePos.setXYZ( 0, 0, 0, 0 );
		linePos.setXYZ( 1, pos.x, pos.y, pos.z );
		linePos.needsUpdate = true;
		lightHelperLine.visible = state.showLightHelper;

	}

	if ( sceneDirLight ) {

		sceneDirLight.position.copy( worldLightDir ).multiplyScalar( 4.0 );
		sceneDirLight.color.set( state.lightColor );
		sceneDirLight.intensity = state.lightIntensity * 1.4;

	}

	if ( sceneAmbLight ) {

		sceneAmbLight.color.set( state.ambientColor );
		sceneAmbLight.intensity = Math.max( 0.35, state.ambientIntensity * 2.2 );

	}

}

function updateTestCube(): void {

	if ( ! testCube ) return;
	testCube.visible = state.showCube;
	testCube.position.set( state.cubeX, state.cubeY, state.cubeZ );
	testCube.scale.setScalar( state.cubeSize );
	testCube.material.color.set( state.cubeColor );

}

function syncUniformsFromState(): void {

	if ( ! shaderUniforms ) return;

	shaderUniforms.shadowColor.value.set( state.shadowColor );
	shaderUniforms.midColor.value.set( state.midColor );
	shaderUniforms.highlightColor.value.set( state.highlightColor );
	shaderUniforms.rimColor.value.set( state.rimColor );
	shaderUniforms.coreGlowColor.value.set( state.coreGlowColor );
	shaderUniforms.ambientColor.value.set( state.ambientColor );

	shaderUniforms.lightColor.value.set( state.lightColor );
	shaderUniforms.lightIntensity.value = state.lightIntensity;
	shaderUniforms.ambientIntensity.value = state.ambientIntensity;
	shaderUniforms.shadowAbsorption.value = state.shadowAbsorption;
	shaderUniforms.shadowDistance.value = state.shadowDistance;
	shaderUniforms.shadowContrast.value = state.shadowContrast;
	shaderUniforms.reliefStrength.value = state.reliefStrength;
	shaderUniforms.powderStrength.value = state.powderStrength;
	shaderUniforms.rimIntensity.value = state.rimIntensity;
	shaderUniforms.anisotropy.value = state.anisotropy;
	shaderUniforms.coreGlowIntensity.value = state.coreGlowIntensity;

	shaderUniforms.threshold.value = state.threshold;
	shaderUniforms.range.value = state.range;
	shaderUniforms.opacity.value = state.opacity;
	shaderUniforms.absorption.value = state.absorption;
	shaderUniforms.steps.value = state.steps;
	shaderUniforms.jitter.value = state.jitter;
	shaderUniforms.exposure.value = state.exposure;
	shaderUniforms.contrast.value = state.contrast;

	shaderUniforms.detailScale.value = state.detailScale;
	shaderUniforms.detailStrength.value = state.detailStrength;
	shaderUniforms.warpStrength.value = state.warpStrength;
	shaderUniforms.warpScale.value = state.warpScale;
	shaderUniforms.mouseDisperse.value = state.mouseDisperse;
	shaderUniforms.mouseTurbulence.value = state.mouseTurbulence;
	shaderUniforms.mouseClampToSmoke.value = state.mouseClampToSmoke ? 1.0 : 0.0;

	if ( cloudMesh ) {

		cloudMesh.scale.setScalar( state.volumeScale );

	}

	updateLightDirection();
	updateSkyGradient();
	updateTestCube();

}

let regenerateTimeout: ReturnType<typeof setTimeout> | null = null;

function scheduleVolumeRegeneration( immediate = false ): void {

	if ( regenerateTimeout ) clearTimeout( regenerateTimeout );

	const run = () => {

		const newData = generateCloudVolumeData( {
			size: 128,
			scale: state.noiseScale,
			octaves: Math.round( state.noiseOctaves ),
			turbulence: state.noiseTurbulence,
			tendrils: state.tendrils,
			falloffPower: state.falloffPower,
			seed: state.seed
		} );

		( cloudTexture.image.data as Uint8Array ).set( newData );
		cloudTexture.needsUpdate = true;

	};

	if ( immediate ) {

		run();

	} else {

		regenerateTimeout = setTimeout( run, 120 );

	}

}

function buildGUI(): void {

	const gui = new GUI( { title: 'Volumetric Smoke Controls' } );

	// Presets
	const presetFolder = gui.addFolder( 'Presets' );
	presetFolder.add( state, 'preset', Object.keys( PRESETS ) ).name( 'Style' ).onChange( ( name: string ) => {

		const p = PRESETS[ name ];
		if ( ! p ) return;
		Object.assign( state, p );
		if ( p.renderScale !== undefined ) {

			renderer.setPixelRatio( state.renderScale );
			renderer.setSize( window.innerWidth, window.innerHeight );

		}
		syncUniformsFromState();
		scheduleVolumeRegeneration( true );
		gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

	} );

	presetFolder.add( {
		copySettings: () => {

			const { preset: _preset, showLightHelper: _helper, ...exportObj } = state;
			void _preset;
			void _helper;
			navigator.clipboard?.writeText( JSON.stringify( exportObj, null, 2 ) );

		}
	}, 'copySettings' ).name( 'Copy Settings JSON' );

	// Cloud / Smoke Colors
	const colorFolder = gui.addFolder( 'Cloud / Smoke Colors' );
	colorFolder.addColor( state, 'shadowColor' ).name( 'Deep Shadow' ).onChange( syncUniformsFromState );
	colorFolder.addColor( state, 'midColor' ).name( 'Base Color' ).onChange( syncUniformsFromState );
	colorFolder.addColor( state, 'highlightColor' ).name( 'Highlight' ).onChange( syncUniformsFromState );
	colorFolder.addColor( state, 'rimColor' ).name( 'Rim Glow' ).onChange( syncUniformsFromState );
	colorFolder.addColor( state, 'coreGlowColor' ).name( 'Core Glow' ).onChange( syncUniformsFromState );
	colorFolder.add( state, 'coreGlowIntensity', 0, 1.5, 0.01 ).name( 'Core Glow Intensity' ).onChange( syncUniformsFromState );
	colorFolder.add( state, 'exposure', 0.4, 2.2, 0.01 ).name( 'Exposure' ).onChange( syncUniformsFromState );
	colorFolder.add( state, 'contrast', 0.6, 2.2, 0.01 ).name( 'Contrast' ).onChange( syncUniformsFromState );

	// Lighting & Shadows
	const lightFolder = gui.addFolder( 'Lighting & Shadows' );
	lightFolder.addColor( state, 'lightColor' ).name( 'Light Color' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'lightIntensity', 0, 3.0, 0.02 ).name( 'Light Intensity' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'lightAzimuth', - 180, 180, 1 ).name( 'Light Azimuth (°)' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'lightElevation', - 90, 90, 1 ).name( 'Light Elevation (°)' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'showLightHelper' ).name( 'Show Light Helper' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'shadowAbsorption', 0, 8.0, 0.05 ).name( 'Shadow Depth' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'shadowDistance', 0.05, 0.8, 0.01 ).name( 'Shadow Distance' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'shadowContrast', 0.5, 2.8, 0.02 ).name( 'Shadow Contrast' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'reliefStrength', 0, 6.0, 0.05 ).name( 'Surface Relief' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'powderStrength', 0, 4.0, 0.05 ).name( 'Powder Darkening' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'rimIntensity', 0, 2.0, 0.02 ).name( 'Rim Intensity' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'anisotropy', - 0.5, 0.85, 0.01 ).name( 'Phase Anisotropy (HG)' ).onChange( syncUniformsFromState );
	lightFolder.addColor( state, 'ambientColor' ).name( 'Ambient Color' ).onChange( syncUniformsFromState );
	lightFolder.add( state, 'ambientIntensity', 0, 1.0, 0.01 ).name( 'Ambient Intensity' ).onChange( syncUniformsFromState );

	// Density & Shape
	const densityFolder = gui.addFolder( 'Density & Shape' );
	densityFolder.add( state, 'threshold', 0.02, 0.65, 0.005 ).name( 'Threshold' ).onChange( syncUniformsFromState );
	densityFolder.add( state, 'range', 0.01, 0.4, 0.005 ).name( 'Edge Softness' ).onChange( syncUniformsFromState );
	densityFolder.add( state, 'opacity', 0.05, 1.5, 0.01 ).name( 'Density' ).onChange( syncUniformsFromState );
	densityFolder.add( state, 'absorption', 0.2, 5.0, 0.05 ).name( 'Absorption' ).onChange( syncUniformsFromState );
	densityFolder.add( state, 'volumeScale', 0.5, 3.0, 0.05 ).name( 'Volume Size (Cube)' ).onChange( syncUniformsFromState );

	// Performance
	const perfFolder = gui.addFolder( 'Performance' );
	perfFolder.add( state, 'renderScale', 0.5, Math.min( window.devicePixelRatio || 2, 2 ), 0.1 ).name( 'Render Scale (DPR)' ).onChange( ( v: number ) => {

		renderer.setPixelRatio( v );
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );
	perfFolder.add( state, 'steps', 24, 160, 1 ).name( 'Raymarch Steps' ).onChange( syncUniformsFromState );
	perfFolder.add( state, 'jitter', 0, 1, 0.05 ).name( 'Ray Jitter (IGN)' ).onChange( syncUniformsFromState );

	// Background
	const bgFolder = gui.addFolder( 'Background Gradient' );
	bgFolder.addColor( state, 'bgTop' ).name( 'Top' ).onChange( updateSkyGradient );
	bgFolder.addColor( state, 'bgMid' ).name( 'Middle' ).onChange( updateSkyGradient );
	bgFolder.addColor( state, 'bgBottom' ).name( 'Bottom' ).onChange( updateSkyGradient );

	// Motion & Real-time Detail
	const motionFolder = gui.addFolder( 'Motion, Warp & Detail' );
	motionFolder.add( state, 'warpStrength', 0, 0.35, 0.005 ).name( 'Domain Warp' ).onChange( syncUniformsFromState );
	motionFolder.add( state, 'warpScale', 0.4, 3.5, 0.05 ).name( 'Warp Scale' ).onChange( syncUniformsFromState );
	motionFolder.add( state, 'detailScale', 0.5, 5.0, 0.05 ).name( 'Detail Scale' ).onChange( syncUniformsFromState );
	motionFolder.add( state, 'detailStrength', 0, 1.0, 0.01 ).name( 'Detail Erosion' ).onChange( syncUniformsFromState );
	motionFolder.add( state, 'windX', - 0.25, 0.25, 0.005 ).name( 'Wind X' );
	motionFolder.add( state, 'windY', - 0.25, 0.25, 0.005 ).name( 'Wind Y (Rise)' );
	motionFolder.add( state, 'windZ', - 0.25, 0.25, 0.005 ).name( 'Wind Z' );
	motionFolder.add( state, 'rotationSpeed', - 0.6, 0.6, 0.01 ).name( 'Rotation Speed' );

	// 3D Noise Volume Generator
	const volumeFolder = gui.addFolder( '3D Noise Generator' );
	volumeFolder.add( state, 'noiseScale', 0.02, 0.12, 0.002 ).name( 'Noise Scale' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( state, 'noiseOctaves', 1, 5, 1 ).name( 'FBM Octaves' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( state, 'noiseTurbulence', 0, 1, 0.02 ).name( 'Billow Turbulence' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( state, 'tendrils', 0, 1.4, 0.02 ).name( 'Tendrils / Branches' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( state, 'falloffPower', 0.8, 3.5, 0.05 ).name( 'Radial Falloff' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( state, 'seed', 0, 100, 0.1 ).name( 'Seed' ).onChange( () => scheduleVolumeRegeneration() );
	volumeFolder.add( {
		randomize: () => {

			state.seed = Number( ( Math.random() * 100 ).toFixed( 1 ) );
			scheduleVolumeRegeneration( true );
			gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

		}
	}, 'randomize' ).name( 'Randomize Volume' );

	// Mouse Fluid Interaction (32^3 painted 3D wake field)
	const mouseFolder = gui.addFolder( 'Mouse Interaction' );
	mouseFolder.add( state, 'mouseEnabled' ).name( 'Enable Hover Wake' ).onChange( ( enabled: boolean ) => {

		if ( ! enabled ) interactionField.clear();

	} );
	mouseFolder.add( state, 'mouseDepthMode', {
		'Front Surface (Dome)': 'dome',
		'Flat Plane (Custom Depth)': 'plane',
		'Full Tunnel (Through)': 'tunnel'
	} ).name( 'Depth Mode' ).onChange( () => {

		hasLastHit = false;

	} );
	mouseFolder.add( state, 'mouseDomeStrength', 0, 1.0, 0.05 ).name( 'Dome Curve (0=Mid)' );
	mouseFolder.add( state, 'mouseDepthOffset', - 0.45, 0.45, 0.01 ).name( 'Depth Offset (Z)' );
	mouseFolder.add( state, 'mouseClampToSmoke' ).name( 'No Ghost Smoke' ).onChange( syncUniformsFromState );
	mouseFolder.add( state, 'mouseRadius', 0.02, 0.25, 0.005 ).name( 'Brush Radius' );
	mouseFolder.add( state, 'mousePush', 0, 0.6, 0.01 ).name( 'Outward Scatter' );
	mouseFolder.add( state, 'mouseDrag', 0, 0.9, 0.01 ).name( 'Swipe Flow' );
	mouseFolder.add( state, 'mouseDisperse', 0, 1.0, 0.02 ).name( 'Center Disperse' ).onChange( syncUniformsFromState );
	mouseFolder.add( state, 'mouseSwirl', 0, 1.8, 0.02 ).name( 'Vortex Swirl' );
	mouseFolder.add( state, 'mouseTurbulence', 0, 0.65, 0.01 ).name( 'Wake Turbulence' ).onChange( syncUniformsFromState );
	mouseFolder.add( state, 'mouseDelay', 0, 0.5, 0.01 ).name( 'Wind Delay (s)' );
	mouseFolder.add( state, 'mouseRecovery', 0.06, 1.5, 0.02 ).name( 'Fade Speed' );

	// Test Cube (for checking volumetric wrapping around 3D models)
	const cubeFolder = gui.addFolder( 'Test Cube' );
	cubeFolder.add( state, 'showCube' ).name( 'Show Cube' ).onChange( updateTestCube );
	cubeFolder.add( state, 'cubeX', - 0.8, 0.8, 0.01 ).name( 'Position X' ).onChange( updateTestCube );
	cubeFolder.add( state, 'cubeY', - 0.8, 0.8, 0.01 ).name( 'Position Y' ).onChange( updateTestCube );
	cubeFolder.add( state, 'cubeZ', - 0.8, 0.8, 0.01 ).name( 'Position Z' ).onChange( updateTestCube );
	cubeFolder.add( state, 'cubeSize', 0.08, 0.6, 0.01 ).name( 'Size' ).onChange( updateTestCube );
	cubeFolder.addColor( state, 'cubeColor' ).name( 'Color' ).onChange( updateTestCube );
	cubeFolder.add( state, 'cubeRotate' ).name( 'Auto Rotate' );

	presetFolder.open();
	mouseFolder.open();
	perfFolder.open();
	cubeFolder.close();
	colorFolder.close();
	lightFolder.close();
	densityFolder.close();
	bgFolder.close();
	motionFolder.close();
	volumeFolder.close();

}

async function init(): Promise<void> {

	renderer = new THREE.WebGPURenderer( { antialias: false } );
	renderer.setPixelRatio( state.renderScale );
	renderer.setSize( window.innerWidth, window.innerHeight );
	document.body.appendChild( renderer.domElement );

	await renderer.init();

	const badge = document.getElementById( 'backend-badge' );
	if ( badge ) {

		const isWebGPU = Boolean( ( renderer.backend as { isWebGPUBackend?: boolean } )?.isWebGPUBackend );
		badge.textContent = isWebGPU ? 'WebGPU' : 'WebGL2 Fallback';

	}

	fpsBadge = document.getElementById( 'fps-badge' );

	scene = new THREE.Scene();

	camera = new THREE.PerspectiveCamera( 55, window.innerWidth / window.innerHeight, 0.1, 100 );
	camera.position.set( 0, 0.14, 2.2 );

	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableDamping = true;
	controls.dampingFactor = 0.06;
	controls.minDistance = 0.7;
	controls.maxDistance = 6.0;

	// Sky dome with editable vertical gradient
	skyCanvas = document.createElement( 'canvas' );
	skyCanvas.width = 2;
	skyCanvas.height = 64;
	const ctx = skyCanvas.getContext( '2d' );
	if ( ! ctx ) throw new Error( 'Failed to get 2D canvas context' );
	skyCtx = ctx;

	skyTexture = new THREE.CanvasTexture( skyCanvas );
	skyTexture.colorSpace = THREE.SRGBColorSpace;
	updateSkyGradient();

	const sky = new THREE.Mesh(
		new THREE.SphereGeometry( 12, 32, 24 ),
		new THREE.MeshBasicNodeMaterial( { map: skyTexture, side: THREE.BackSide, depthWrite: false } )
	);
	scene.add( sky );

	// Scene lights synced with the volumetric smoke light
	sceneDirLight = new THREE.DirectionalLight( state.lightColor, state.lightIntensity * 1.4 );
	scene.add( sceneDirLight );

	sceneAmbLight = new THREE.AmbientLight( state.ambientColor, Math.max( 0.35, state.ambientIntensity * 2.2 ) );
	scene.add( sceneAmbLight );

	// Test cube sitting inside the smoke to demonstrate depth-aware wrapping
	const cubeMat = new THREE.MeshStandardNodeMaterial( {
		color: state.cubeColor,
		roughness: 0.35,
		metalness: 0.25
	} );
	testCube = new THREE.Mesh( new THREE.BoxGeometry( 1, 1, 1 ), cubeMat );
	testCube.rotation.set( 0.45, 0.65, 0.2 );
	scene.add( testCube );

	// Light direction helper (optional toggle)
	lightHelperGroup = new THREE.Mesh(
		new THREE.SphereGeometry( 0.045, 16, 16 ),
		new THREE.MeshBasicNodeMaterial( { color: 0xffd4e0 } )
	);
	lightHelperGroup.visible = false;
	scene.add( lightHelperGroup );

	const lineGeo = new THREE.BufferGeometry().setFromPoints( [
		new THREE.Vector3( 0, 0, 0 ),
		new THREE.Vector3( 0.5, 0.7, 0.5 )
	] );
	lightHelperLine = new THREE.Line(
		lineGeo,
		new THREE.LineBasicMaterial( { color: 0xff6b9a, transparent: true, opacity: 0.5 } )
	);
	lightHelperLine.visible = false;
	scene.add( lightHelperLine );

	// Generate 3D volume textures
	cloudTexture = createCloudVolumeTexture( {
		size: 128,
		scale: state.noiseScale,
		octaves: state.noiseOctaves,
		turbulence: state.noiseTurbulence,
		tendrils: state.tendrils,
		falloffPower: state.falloffPower,
		seed: state.seed
	} );
	detailTexture = createDetailVolumeTexture( 64 );

	// Volumetric smoke material & mesh
	const { material, uniforms } = createVolumetricCloudMaterial(
		cloudTexture,
		detailTexture,
		interactionField.texture,
		state
	);
	shaderUniforms = uniforms;

	cloudMesh = new THREE.Mesh( new THREE.BoxGeometry( 1, 1, 1 ), material );
	scene.add( cloudMesh );

	syncUniformsFromState();
	buildGUI();

	window.addEventListener( 'resize', onWindowResize );
	renderer.domElement.addEventListener( 'pointerdown', () => {

		isPointerDown = true;
		hasLastHit = false;

	} );
	window.addEventListener( 'pointerup', () => {

		isPointerDown = false;
		hasLastHit = false;

	} );
	renderer.domElement.addEventListener( 'pointerleave', () => {

		hasLastHit = false;

	} );
	renderer.domElement.addEventListener( 'pointermove', onPointerMove );

	renderer.setAnimationLoop( animate );

}

function onPointerMove( event: PointerEvent ): void {

	if ( ! state.mouseEnabled || isPointerDown || ! cloudMesh || ! camera ) return;

	pointerNdc.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	pointerNdc.y = - ( event.clientY / window.innerHeight ) * 2 + 1;

	// Intersect pointer ray with a camera-facing plane through the cloud center
	planeNormal.copy( camera.position ).sub( cloudMesh.position ).normalize();
	hitPlane.setFromNormalAndCoplanarPoint( planeNormal, cloudMesh.position );
	raycaster.setFromCamera( pointerNdc, camera );

	if ( ! raycaster.ray.intersectPlane( hitPlane, worldHit ) ) return;

	localHit.copy( worldHit );
	cloudMesh.worldToLocal( localHit );

	// Only disturb smoke when hovering near/inside the cubic volume
	if ( Math.abs( localHit.x ) > 0.72 || Math.abs( localHit.y ) > 0.72 || Math.abs( localHit.z ) > 0.72 ) {

		hasLastHit = false;
		return;

	}

	// Camera-facing direction in cloudMesh local space (from center toward camera)
	invQuat.copy( cloudMesh.quaternion ).invert();
	localCamDir.copy( planeNormal ).applyQuaternion( invQuat ).normalize();

	// Apply selected depth surface:
	// - 'dome': front hemisphere (front surface near center, smoothly curving to mid-plane at edges)
	// - 'plane': flat camera-facing plane shifted by mouseDepthOffset
	// - 'tunnel': stamps at multiple depths through the cloud along the camera ray
	if ( state.mouseDepthMode === 'dome' ) {

		const domeR = 0.34;
		const r2 = localHit.lengthSq();
		const domeZ = ( r2 < domeR * domeR ? Math.sqrt( domeR * domeR - r2 ) : 0 ) * state.mouseDomeStrength;
		localHit.addScaledVector( localCamDir, domeZ + state.mouseDepthOffset );

	} else if ( state.mouseDepthMode === 'plane' ) {

		localHit.addScaledVector( localCamDir, state.mouseDepthOffset );

	} else {

		localHit.addScaledVector( localCamDir, state.mouseDepthOffset );

	}

	if ( ! hasLastHit ) {

		lastLocalHit.copy( localHit );
		hasLastHit = true;
		return;

	}

	if ( state.mouseDelay <= 0.005 ) {

		applyWindStroke( lastLocalHit, localHit, localCamDir, state.mouseDepthMode );

	} else {

		// Queue gust to arrive after mouseDelay seconds for a realistic wind-wave lag
		pendingWindQueue.push( {
			from: lastLocalHit.clone(),
			to: localHit.clone(),
			camDir: localCamDir.clone(),
			mode: state.mouseDepthMode,
			readyAt: performance.now() + state.mouseDelay * 1000
		} );

	}

	lastLocalHit.copy( localHit );

}

function applyWindStroke(
	from: THREE.Vector3,
	to: THREE.Vector3,
	camDir: THREE.Vector3,
	mode: CloudState['mouseDepthMode']
): void {

	const stampParams = {
		radius: state.mouseRadius,
		push: state.mousePush,
		drag: state.mouseDrag,
		swirl: state.mouseSwirl,
		disperse: state.mouseDisperse
	};

	if ( mode === 'tunnel' ) {

		const layers = [ - 0.24, 0.0, 0.24 ];
		for ( let l = 0; l < layers.length; l ++ ) {

			tunnelFrom.copy( from ).addScaledVector( camDir, layers[ l ] );
			tunnelTo.copy( to ).addScaledVector( camDir, layers[ l ] );
			interactionField.paintStroke( tunnelFrom, tunnelTo, stampParams );

		}

	} else {

		interactionField.paintStroke( from, to, stampParams );

	}

}

function onWindowResize(): void {

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();
	renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate(): void {

	const dt = Math.min( clock.getDelta(), 0.05 );

	fpsFrames ++;
	const now = performance.now();
	const elapsed = now - fpsLastTime;
	if ( elapsed >= 250 ) {

		const fps = Math.round( ( fpsFrames * 1000 ) / elapsed );
		const ms = ( elapsed / fpsFrames ).toFixed( 1 );
		if ( fpsBadge ) fpsBadge.textContent = `${ fps } FPS (${ ms } ms)`;
		fpsFrames = 0;
		fpsLastTime = now;

	}

	controls.update();

	// Rotate cloud mesh
	cloudMesh.rotation.y -= dt * state.rotationSpeed;

	// Gently spin test cube
	if ( testCube && state.cubeRotate ) {

		testCube.rotation.y += dt * 0.35;
		testCube.rotation.x += dt * 0.18;

	}

	// Transform world light direction into cloudMesh's local space so shadows stay anchored to the light
	if ( shaderUniforms ) {

		invQuat.copy( cloudMesh.quaternion ).invert();
		localLightDir.copy( worldLightDir ).applyQuaternion( invQuat ).normalize();
		shaderUniforms.lightDir.value.copy( localLightDir );

		// Scroll 3D detail noise for continuous billowing smoke motion
		shaderUniforms.windOffset.value.x -= dt * state.windX;
		shaderUniforms.windOffset.value.y -= dt * state.windY;
		shaderUniforms.windOffset.value.z -= dt * state.windZ;

		// Flush delayed wind gusts whose travel time has elapsed
		if ( pendingWindQueue.length > 0 ) {

			if ( ! state.mouseEnabled ) {

				pendingWindQueue.length = 0;

			} else {

				let consumed = 0;
				while ( consumed < pendingWindQueue.length && pendingWindQueue[ consumed ].readyAt <= now ) {

					const gust = pendingWindQueue[ consumed ];
					applyWindStroke( gust.from, gust.to, gust.camDir, gust.mode );
					consumed ++;

				}
				if ( consumed > 0 ) pendingWindQueue.splice( 0, consumed );

			}

		}

		// Relax & upload 32^3 painted 3D wake field only when active
		interactionField.update( dt, state.mouseRecovery );
		shaderUniforms.mouseActive.value = state.mouseEnabled && interactionField.isActive ? interactionField.activityLevel : 0;

	}

	renderer.render( scene, camera );

}

void init();
