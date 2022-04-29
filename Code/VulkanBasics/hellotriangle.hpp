#pragma once
#pragma region
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>     //For Vulkan-supported glfw headers

#include <iostream>
#include<stdexcept>
#include<cstdlib>
#pragma endregion


class HelloTriangleApplication {
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();

	}

private:
	//put the process methods in the private scope
	void initWindow()
	{
		glfwInit();     //initialize glfw libraries before using glfw functions

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   //use hint to set window properties, no link to OPENGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  //NO RESIZABLE Window

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}
	void initVulkan()
	{

	}

	void mainLoop()
	{
		//KEEP THE Window until close it
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();  //process all the window-related events
		}
	}

	void cleanup()
	{
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	GLFWwindow* window;
	const uint32_t WIDTH = 800;
	const uint32_t HEIGHT = 600;
};
