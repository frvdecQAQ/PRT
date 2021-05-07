#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
class shRotate{
public:
	shRotate(int band);
	~shRotate();
	void rotate(const float* coef_in,
		float* coef_out,
		const glm::mat4& m,
		int band);
private:
	const float eps = 1e-6;
	float* pre_cos, * pre_sin, * coef_tmp;
	int SHIndex(int l, int m);
	void toZYZ(const glm::mat4& m, float& alpha, float& beta, float& gamma);
	void SHRotateZ(const float* c_in, float* c_out, float alpha, int lmax);
	void SHRotateXPlus(const float* coef_in, float* coef_out, int lmax);
	void SHRotateXMinus(const float* coef_in, float* coef_out, int lmax);
	void sinCosIndexed(float s, float c, int n);
};

