#version 460
#extension GL_EXT_debug_printf : require
//for pass in vec3 storage buffer
#extension GL_EXT_scalar_block_layout : require
//for ray-tracing
#extension GL_EXT_ray_query:require


layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

//from the binding of descriptor set
//this information is used for binding 0 and descriptor set 0
//scalr for pack the elements consecutively in memory instead of 4-float-point element way
layout(binding = 0, set = 0, scalar) buffer storageBuffer
{
	vec3 imageData[];
};

// uniform means only read
layout(binding=1, set=0) uniform accelerationStructureEXT tlas;

//Access to the vertex and index buffer
layout(binding = 2, set = 0, scalar) buffer Vertices
{
	vec3 vertices[];
};

layout(binding = 3, set = 0, scalar) buffer Indices
{
	uint indices[];
};

//is dark gray below the horizon;
//is white near the horizon;
//fades to a blue near the zenith.
vec3 skyColor(vec3 direction)
{
	//+y in world space is up	
	if(direction.y > 0.0f)
	{
		return mix(vec3(1.0f), vec3(0.25f, 0.5f, 1.0f), direction.y);
	}
	else   //too down
	{
		return vec3(0.03f);
	}
	
}

//Random number generation using pcg32i_random_t
uint stepRNG(uint rngState)
{
	return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
	// Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
	rngState = stepRNG(rngState);
	uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
	word = (word >> 22) ^ word;
	return float(word) / 4294967295.0f;
}


//define a custom struct for hit info
struct HitInfo
{
	vec3 color;
	vec3 worldPosition;
	vec3 worldNormal;
};

HitInfo getObjectHitInfo (rayQueryEXT rayQuery)
{
	HitInfo result;

	//Get the ID of the triangle
	const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
	const uint i0 = indices[3 * primitiveID];  //from index buffer
	const uint i1 = indices[3 * primitiveID + 1];  //from index buffer
	const uint i2 = indices[3 * primitiveID + 2];  //from index buffer

	//from vertice buffer
	const vec3 v0 = vertices[i0];
	const vec3 v1 = vertices[i1];
	const vec3 v2 = vertices[i2];


	// utilize front-face triangels (counter-clock winding order) to determine the sole normal
	const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
	// For the main tutorial, object space is the same as world space: (No transformation involved)
    result.worldNormal = objectNormal;

	//Get the barycentric coordinates of the intersection
	vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
	barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

	//compute the position of the intersection
	const vec3 intersectionPoint = barycentrics.x * v0 + barycentrics.y * v1 + barycentrics.z * v2;
	result.worldPosition = intersectionPoint;
	//vec3(0.7, 0.7, 0.7);   (gray color)
	//normal way,  vec3(0.5) + 0.5 * result.worldNormal; 
	//simulate cornelbox
	result.color = vec3(0.8f);
	const float dotX = dot(result.worldNormal, vec3(1.0, 0.0, 0.0));
	if (dotX > 0.99)
	{
		result.color = vec3(0.8, 0.0, 0.0);
	}
	else if (dotX < -0.99)
	{
		result.color = vec3(0.0, 0.8, 0.0);
	}
	return result;
}

void main()
{
	//Resolution of the buffer, hardcoded
	const uvec2 resolution = uvec2(800, 600);

	
	// get the coordinates of the pixel for this invocation
	const uvec2 pixel = gl_GlobalInvocationID.xy;

	//if the pixel is outside of the image, don't do anything
	if ((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
	{
		return ;
	}

	//conform the vertical field of view
	const vec3 cameraOrigin = vec3(-0.001, 1.0, 6.0);

	// The sum of the colors of all of the samples.
	vec3 summedPixelColor = vec3(0.0);
	
	const int NUM_SAMPLES = 64;
	uint rngState = resolution.x * pixel.y + pixel.x;   //for pseudo-random numbers

	for (int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
	{
		// Rays always originate at the camera for now. In the future, they'll
		// bounce around the scene.
		vec3 rayOrigin = cameraOrigin;

	  // Compute the direction of the ray for this pixel. To do this, we first
	  // transform the screen coordinates to look like this, where a is the
	  // aspect ratio (width/height) of the screen:
	  //           1
	  //    .------+------.
	  //    |      |      |
	  // -a + ---- 0 ---- + a
	  //    |      |      |
	  //    '------+------'
	  //          -1
	  // vertical field of view

	  //flip y-axis, for y-axis up for 3d case
	  const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
	  const vec2 screenUV = vec2((2.0 * float(randomPixelCenter.x)+1.0 - resolution.x)/resolution.y, -(2.0* float(randomPixelCenter.y)+1.0 -resolution.y)/resolution.y);

	  // define the field of view by the vertical slope of the topmost rays
	  //and create a ray direction
	  const float fovVerticalSlope = 1.0/5.0;
	  vec3 rayDirection = vec3(fovVerticalSlope*screenUV.x, fovVerticalSlope * screenUV.y, -1.0);   //into the screen (-1.0)
	  rayDirection = normalize(rayDirection);  //normalize the direction


	  vec3 accumulatedRayColor = vec3(1.0);  //The amount of light that made it to the end of the current ray
	  //vec3 pixelColor = vec3(0.0);  //initialized with values for avoiding unwanted values

	  //begin the raytracing loop (at most 32 times, 31 bounces)
	  for (int tracedSegments = 0; tracedSegments < 32; tracedSegments++)
	  {
		  //Trace the ray and see if and where it intersects the scene!
		  //First, initialize a ray query object;
		  rayQueryEXT rayQuery;
		  rayQueryInitializeEXT(rayQuery,  //Ray query
			  tlas,   //top-level structure
			  gl_RayFlagsOpaqueEXT,  // Ray flags, force iterating over every intersection (note: slow)
			  0xFF,
			  rayOrigin,
			  0.0,      //minimum t-value
			  rayDirection,
			  10000.0);   //Maximum t-value

		//start traversel and loop over all ray-scene intersections
		//rayQuery stores a "committed" intersection, the closest intersection (if any).
		  while (rayQueryProceedEXT(rayQuery))
		  {

		  }

		  //Check whether it hit a triangle / sky
		  if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)  //check intersection type
		  {
			  //find the hit info
			  HitInfo hitInfo = getObjectHitInfo(rayQuery);

			  //Apply the color absorption
			  accumulatedRayColor *= hitInfo.color;

			  //Update ray for the newly bounce
			  //Flip the normal so it points against the ray direction
			  hitInfo.worldNormal = faceforward(hitInfo.worldNormal, rayDirection, hitInfo.worldNormal);

			  rayOrigin = hitInfo.worldPosition + 0.0001 * hitInfo.worldNormal;

			  //lambertian way
			  const float theta = 6.2831853 * stepAndOutputRNGFloat(rngState);
			  const float u =  2 * stepAndOutputRNGFloat(rngState) -1.0;
			  const float r = sqrt(1.0 - u*u);

			  rayDirection = hitInfo.worldNormal + vec3(r*cos(theta), r*sin(theta),u);
			  //pure specular way
			  //rayDirection = reflect(rayDirection, hitInfo.worldNormal);
		  }
		  else
		  {
			  // reintepreted as the sky color
			  summedPixelColor += clamp(accumulatedRayColor * skyColor(rayDirection),0.0, 1.0);
			  break;
		  }

	  }

    }
  // Get the t-value of the intersection (if there's no intersection, this will
  // be tMax = 10000.0). "true" says "get the committed intersection."
  //const float t = rayQueryGetIntersectionTEXT(rayQuery, true);

  summedPixelColor = summedPixelColor / float(NUM_SAMPLES);  //anti-aliased results
  //Get the index of this invocation in the buffer
  uint linearIndex = resolution.x * pixel.y + pixel.x;
  //Give the pixel color, apply the gamma correction as well
  imageData[linearIndex] = vec3(pow(summedPixelColor.r, 1.0/2.2), pow(summedPixelColor.g, 1.0/2.2), pow(summedPixelColor.b, 1.0/2.2));


  //visualize random number generations
  //uint rngState = resolution.x * pixel.y + pixel.x;
  //pixelColor = vec3(stepAndOutputRNGFloat(rngState));
  //imageData[linearIndex] = vec3(pow(pixelColor.r, 1.0 / 2.2), pow(pixelColor.g, 1.0 / 2.2), pow(pixelColor.b, 1.0 / 2.2));


}