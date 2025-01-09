// code for depth-of-field, mouse + keyboard user interaction based on https://github.com/peterkutz/GPUPathTracer

#pragma once
#include "Camera.h"


inline bool buffer_reset = false;


/**
 * Creats new interactive camera at the interactiveCamera parameter
 * effectively resetting it to the default parameters
 */
inline void initCamera(InteractiveCamera& interactiveCamera, int scrwidth,int scrheight)
{
	//Resets the camera to a new initial state
	interactiveCamera = InteractiveCamera( scrwidth, scrheight);

	interactiveCamera.setResolution(scrwidth, scrheight);
	interactiveCamera.setFOVX(45);
}

// mouse event handlers
inline int lastX = 0, lastY = 0;
inline int theButtonState = 0;
inline int theModifierState = 0;

// keyboard interaction
inline void keyboard(InteractiveCamera& interactiveCamera, unsigned char key, int /*x*/, int /*y*/, int scrwidth,int scrheight)
{
	switch (key) {

	case(27) : exit(0);
	case(' ') : initCamera(interactiveCamera, scrwidth, scrheight); buffer_reset = true; break;
	case('a') : interactiveCamera.strafe(-0.05f); buffer_reset = true; break;
	case('d') : interactiveCamera.strafe(0.05f); buffer_reset = true; break;
	case('r') : interactiveCamera.changeAltitude(0.05f); buffer_reset = true; break;
	case('f') : interactiveCamera.changeAltitude(-0.05f); buffer_reset = true; break;
	case('w') : interactiveCamera.goForward(0.05f); buffer_reset = true; break;
	case('s') : interactiveCamera.goForward(-0.05f); buffer_reset = true; break;
	case('g') : interactiveCamera.changeApertureDiameter(0.1); buffer_reset = true; break;
	case('h') : interactiveCamera.changeApertureDiameter(-0.1); buffer_reset = true; break;
	case('t') : interactiveCamera.changeFocalDistance(0.1); buffer_reset = true; break;
	case('y') : interactiveCamera.changeFocalDistance(-0.1); buffer_reset = true; break;
	}
}

inline void specialkeys(InteractiveCamera& interactiveCamera, int key, int, int){

	switch (key) {

	case GLUT_KEY_LEFT:  interactiveCamera.changeYaw(   0.02f); buffer_reset = true; break;
	case GLUT_KEY_RIGHT: interactiveCamera.changeYaw(  -0.02f); buffer_reset = true; break;
	case GLUT_KEY_UP:    interactiveCamera.changePitch( 0.02f); buffer_reset = true; break;
	case GLUT_KEY_DOWN:  interactiveCamera.changePitch(-0.02f); buffer_reset = true; break;

	}
}

// camera mouse controls in X and Y direction
inline void motion(InteractiveCamera& interactiveCamera, int x, int y)
{
	int deltaX = lastX - x;
	int deltaY = lastY - y;

	if (deltaX != 0 || deltaY != 0) {

		if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
		{
			interactiveCamera.changeYaw(deltaX * 0.01);
			interactiveCamera.changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
		{
			interactiveCamera.changeAltitude(-deltaY * 0.01);
		}

		if (theButtonState == GLUT_RIGHT_BUTTON) // camera move
		{
			interactiveCamera.changeRadius(-deltaY * 0.01);
		}

		lastX = x;
		lastY = y;
		buffer_reset = true;
		glutPostRedisplay();

	}
}

inline void zoom(InteractiveCamera& interactiveCamera, const float zoom) {
	interactiveCamera.changeFocalDistance(zoom);
		buffer_reset = true;
}

inline void mouse(InteractiveCamera& interactiveCamera, const int button, const int state, const int x, int const y)
{
	theButtonState = button;
	theModifierState = glutGetModifiers();
	lastX = x;
	lastY = y;
	if (button == 3) // UP
		zoom(interactiveCamera, 0.1f );
	else if (button == 4) // DOWN
		zoom(interactiveCamera, -0.1f );
	motion(interactiveCamera, x, y);
}
