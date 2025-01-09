// code for depth-of-field, mouse + keyboard user interaction based on https://github.com/peterkutz/GPUPathTracer
#include <cmath>
#include "Camera.h"
#include <numbers>

InteractiveCamera::InteractiveCamera(const int scrwidth, const int scrheight)
{
	centerPosition = Vec3f(0, 0, 0);
	yaw = 0;
	pitch = 0.3;
	radius = 4;
	apertureRadius = 0.04; 
	focalDistance = 4.0f;

	resolution = Vec2f(static_cast<float>(scrwidth), static_cast<float>(scrheight));
	fov = Vec2f(40, 40);
}

InteractiveCamera::~InteractiveCamera() = default;

void InteractiveCamera::changeYaw(const float m){
	yaw += m;
	fixYaw();
}

void InteractiveCamera::changePitch(const float m){
	pitch += m;
	fixPitch();
}

void InteractiveCamera::changeRadius(const float m){
	radius += radius * m; // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
	fixRadius();
}

void InteractiveCamera::changeAltitude(const float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void InteractiveCamera::goForward(const float m){
	centerPosition += viewDirection * m;
}

void InteractiveCamera::strafe(const float m){
	Vec3f strafeAxis = cross(viewDirection, Vec3f(0, 1, 0));
	strafeAxis.normalize();
	centerPosition += strafeAxis * m;
}

void InteractiveCamera::rotateRight(const float m){
	float yaw2 = yaw;
	yaw2 += m;
	const float pitch2 = pitch;
	const float xDirection = sin(yaw2) * cos(pitch2);
	const float yDirection = sin(pitch2);
	const float zDirection = cos(yaw2) * cos(pitch2);
	const Vec3f directionToCamera(xDirection, yDirection, zDirection);
	viewDirection = directionToCamera * (-1.0);
}

void InteractiveCamera::changeApertureDiameter(const float m){
	apertureRadius += static_cast<float>((apertureRadius + 0.01) * m); // Change proportional to current apertureRadius.
	fixApertureRadius();
}


void InteractiveCamera::changeFocalDistance(const float m){
	focalDistance += m;
	fixFocalDistance();
}


void InteractiveCamera::setResolution(const float x, const float y){
	resolution = Vec2f(x, y);
	setFOVX(fov.x);
}

float radiansToDegrees(const float radians) {
	const auto degrees = static_cast<float>(radians * 180.0 / std::numbers::pi);
	return degrees;
}

float degreesToRadians(const float degrees) {
	const auto radians = static_cast<float>(degrees / 180.0 * std::numbers::pi);
	return radians;
}

void InteractiveCamera::setFOVX(const float fovx){
	fov.x = fovx;
	fov.y = radiansToDegrees(static_cast<float>(std::atan(tan(degreesToRadians(fovx) * 0.5) * (resolution.y / resolution.x)) * 2.0));
	// resolution float division
}

void InteractiveCamera::buildRenderCamera(Camera& renderCamera){
	const float xDirection = std::sin(yaw) * std::cos(pitch);
	const float yDirection = std::sin(pitch);
	const float zDirection = std::cos(yaw) * std::cos(pitch);
	const Vec3f directionToCamera(xDirection, yDirection, zDirection);
	viewDirection = directionToCamera * (-1.0);
	const Vec3f eyePosition = centerPosition + directionToCamera * radius;
	//Vec3f eyePosition = centerPosition; // rotate camera from stationary viewpoint


	renderCamera.position = eyePosition;
	renderCamera.view = viewDirection;
	renderCamera.up = Vec3f(0, 1, 0);
	renderCamera.resolution = Vec2f(resolution.x, resolution.y);
	renderCamera.fov = Vec2f(fov.x, fov.y);
	renderCamera.apertureRadius = apertureRadius;
	renderCamera.focalDistance = focalDistance;
}

float mod(const float x, const float y) { // Does this account for -y ???
	return x - y * floorf(x / y);
}

void InteractiveCamera::fixYaw() {
    yaw = mod(yaw, 2 * std::numbers::pi); // Normalize the yaw.
}

float clamp2(float n, const float low, const float high) {
	n = fminf(n, high);
	n = fmaxf(n, low);
	return n;
}

void InteractiveCamera::fixPitch() {
	constexpr float padding = 0.05;
	pitch = clamp2(pitch, -(std::numbers::pi / 2) + padding, (std::numbers::pi / 2) - padding); // Limit the pitch.
}

void InteractiveCamera::fixRadius() {
	constexpr float minRadius = 0.2;
	constexpr float maxRadius = 100.0;
	radius = clamp2(radius, minRadius, maxRadius);
}

void InteractiveCamera::fixApertureRadius() {
	constexpr float minApertureRadius = 0.0;
	constexpr float maxApertureRadius = 25.0;
	apertureRadius = clamp2(apertureRadius, minApertureRadius, maxApertureRadius);
}

void InteractiveCamera::fixFocalDistance() {
	constexpr float minFocalDist = 0.2;
	constexpr float maxFocalDist = 100.0;
	focalDistance = clamp2(focalDistance, minFocalDist, maxFocalDist);
}

