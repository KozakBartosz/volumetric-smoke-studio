export interface CloudPreset {
	// Optional performance & cube overrides
	renderScale?: number;
	jitter?: number;
	showCube?: boolean;
	cubeX?: number;
	cubeY?: number;
	cubeZ?: number;
	cubeSize?: number;
	cubeColor?: string;
	cubeRotate?: boolean;

	// Colors
	shadowColor: string;
	midColor: string;
	highlightColor: string;
	rimColor: string;
	coreGlowColor: string;
	ambientColor: string;

	// Lighting & Shadows
	lightColor: string;
	lightIntensity: number;
	lightAzimuth: number;
	lightElevation: number;
	ambientIntensity: number;
	shadowAbsorption: number;
	shadowDistance: number;
	shadowContrast: number;
	reliefStrength: number;
	powderStrength: number;
	rimIntensity: number;
	anisotropy: number;
	coreGlowIntensity: number;

	// Density & Shape
	threshold: number;
	range: number;
	opacity: number;
	absorption: number;
	steps: number;
	exposure: number;
	contrast: number;

	// Detail & Motion
	detailScale: number;
	detailStrength: number;
	warpStrength: number;
	warpScale: number;
	windX: number;
	windY: number;
	windZ: number;
	rotationSpeed: number;
	volumeScale: number;

	// Background
	bgTop: string;
	bgMid: string;
	bgBottom: string;

	// Volume generator
	noiseScale: number;
	noiseOctaves: number;
	noiseTurbulence: number;
	tendrils: number;
	falloffPower: number;
	seed: number;
}

export interface CloudState extends CloudPreset {
	preset: string;
	showLightHelper: boolean;
	renderScale: number;
	jitter: number;
	showCube: boolean;
	cubeX: number;
	cubeY: number;
	cubeZ: number;
	cubeSize: number;
	cubeColor: string;
	cubeRotate: boolean;
}

export interface CloudVolumeOptions {
	size?: number;
	scale?: number;
	octaves?: number;
	lacunarity?: number;
	gain?: number;
	turbulence?: number;
	falloffPower?: number;
	tendrils?: number;
	seed?: number;
}
