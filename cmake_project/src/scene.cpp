#include "scene.h"

void Scene::init(std::string& path) {
	FILE* config_file = fopen((path+"/config.txt").c_str(), "r");
	assert(config_file != NULL);
	fscanf(config_file, "%d", &obj_num);
	char* obj_name = new char[256];
	for (int i = 0; i < obj_num; ++i) {
		fscanf(config_file, "%s", obj_name);
		name_list.push_back(std::string(obj_name));
	}
	int type;
	for (int i = 0; i < obj_num; ++i) {
		fscanf(config_file, "%d", &type);
		std::cout << "type = " << type << std::endl;
		type_list.push_back(type);
	}
	float x, y, z;
	for (int i = 0; i < obj_num; ++i) {
		fscanf(config_file, "%f%f%f", &x, &y, &z);
		scale.push_back(glm::vec3(x, y, z));
		std::cout << x << ' ' << y << ' ' << z << std::endl;
	}
	for (int i = 0; i < obj_num; ++i) {
		fscanf(config_file, "%f%f%f", &x, &y, &z);
		center.push_back(glm::vec3(x, y, z));
		std::cout << x << ' ' << y << ' ' << z << std::endl;
	}
	delete[] obj_name;
	fclose(config_file);
}

void Scene::prepare() {
	//calculate vertex and face
	for (int i = 0; i < obj_num; ++i) {
		vertice_num += (obj_list[i]->_vertices.size() / 3);
		indices_num += (obj_list[i]->_indices.size() / 3);
	}

	//translate to the center
	for (int i = 0; i < obj_num; ++i) {
		glm::vec3 now_center = glm::vec3(obj_list[i]->_cx, obj_list[i]->_cy, obj_list[i]->_cz);
		translate(obj_list[i], center[i]-now_center);
	}
}

void Scene::prepareData(int cur_band, int target_vertex) {
	std::cout << "obj_num = " << obj_num << std::endl;
	int band2 = cur_band * cur_band;
	BVHTree* bvht = new BVHTree[obj_num];
	for (int i = 0; i < obj_num; ++i) {
		bvht[i].build(*obj_list[i]);
	}
	Sampler shadow_sample(98);
	shadow_sample.computeSH(cur_band);

	for (int obj_id = 0; obj_id < obj_num; ++obj_id) {
		Object* obj_now = obj_list[obj_id];
		int vertex_num = obj_now->_vertices.size() / 3;

		FILE* fp = fopen(("./data/"+name_list[obj_id]+"_first_data.txt").c_str(), "w");
		fprintf(fp, "%d %d\n", vertex_num, cur_band);
		std::vector<std::vector<float>>coef;
		std::vector<std::vector<float>>*
			coef_single = new std::vector<std::vector<float>>[obj_num];
		std::vector<float> coef_tmp(band2, 0.0f);
		coef.resize(vertex_num, coef_tmp);
		for (int i = 0; i < obj_num; ++i)coef_single[i].resize(vertex_num, coef_tmp);
		int sample_sz = shadow_sample._samples.size();
		float weight = 4.0f * M_PI / sample_sz;

		std::cout << "task start" << std::endl;

#pragma omp parallel for
		for (int i = 0; i < vertex_num; ++i) {
			if (target_vertex != -1 && i != target_vertex)continue;
			if (i % 10 == 0)std::cout << i << std::endl;
			glm::vec3 now_point =
				glm::vec3(obj_now->_vertices[i * 3], obj_now->_vertices[i * 3 + 1], obj_now->_vertices[i * 3 + 2]);

			glm::vec3 normal =
				glm::vec3(obj_now->_normals[i * 3], obj_now->_normals[i * 3 + 1], obj_now->_normals[i * 3 + 2]);

			normal = glm::normalize(normal);
			glm::vec3 u;
			u = glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f));
			if (glm::dot(u, u) < 1e-3f) {
				u = glm::cross(normal, glm::vec3(1.0f, 0.0f, 0.0f));
			}
			u = glm::normalize(u);
			glm::vec3 v = glm::cross(normal, u);

			//assert(fabs(glm::length(u) - 1) < 1e-6);
			//assert(fabs(glm::length(v) - 1) < 1e-6);
			//assert(fabs(glm::length(normal) - 1) < 1e-6);

			for (int j = 0; j < sample_sz; ++j) {
				bool flag = true;
				bool test_result;
				Sample& stemp = shadow_sample._samples[j];
				glm::vec3 dir_world =
					stemp._cartesCoord[0] * u +
					stemp._cartesCoord[1] * v +
					stemp._cartesCoord[2] * normal;
				Ray testRay(now_point, dir_world);

				for (int k = 0; k < obj_num; ++k) {
					if (stemp._cartesCoord[2] > 0)test_result = (!bvht[k].intersect(testRay, true));
					else test_result = false;
					flag &= test_result;
					if (test_result) {
						for (int t = 0; t < band2; ++t) {
							coef_single[k][i][t] += stemp._SHvalue[t];
						}
					}
				}
				if (flag) {
					for (int k = 0; k < band2; ++k) {
						coef[i][k] += stemp._SHvalue[k];
					}
				}
			}
			for (int j = 0; j < band2; ++j) {
				coef[i][j] *= weight;
				for (int k = 0; k < obj_num; ++k) {
					coef_single[k][i][j] *= weight;
				}
			}
		}
		for (int i = 0; i < vertex_num; ++i) {
			if (target_vertex != -1 && i != target_vertex)continue;
			for (int j = 0; j < band2; ++j) {
				fprintf(fp, "%.10lf\n", coef[i][j]);
			}
		}
		for (int k = 0; k < obj_num; ++k) {
			FILE* fp_tmp = fopen(("./data/"+name_list[obj_id]+"_second_" +name_list[k]+ ".txt").c_str(), "w");
			fprintf(fp_tmp, "%d %d\n", vertex_num, cur_band);
			for (int i = 0; i < vertex_num; ++i) {
				if (target_vertex != -1 && i != target_vertex)continue;
				for (int j = 0; j < band2; ++j) {
					fprintf(fp_tmp, "%.10f\n", coef_single[k][i][j]);
				}
			}
			fclose(fp_tmp);
		}
		fclose(fp);
		delete[] coef_single;
	}
	delete[] bvht;
}

bool Scene::change(glm::vec3& c_pos, glm::vec3& c_dir) {
	glm::mat4 rotateMatrix;
	rotateMatrix[1][1] = std::cos(M_PI / 180);
	rotateMatrix[1][2] = -std::sin(M_PI / 180);
	rotateMatrix[2][1] = std::sin(M_PI / 180);
	rotateMatrix[2][2] = std::cos(M_PI / 180);
	rotateMatrix[0][0] = 1;
	glm::vec3 tmp = c_pos;
	c_pos[0] = rotateMatrix[0][0] * tmp[0] + rotateMatrix[0][1] * tmp[1] + rotateMatrix[0][2] * tmp[2];
	c_pos[1] = rotateMatrix[1][0] * tmp[0] + rotateMatrix[1][1] * tmp[1] + rotateMatrix[1][2] * tmp[2];
	c_pos[2] = rotateMatrix[2][0] * tmp[0] + rotateMatrix[2][1] * tmp[1] + rotateMatrix[2][2] * tmp[2];
	return true;
}

void Scene::translate(Object* obj, glm::vec3 trans_vec) {
	int sz = obj->_vertices.size();
	for (int i = 0; i < sz; ++i) {
		obj->_vertices[i] += trans_vec[i % 3];
		std::cout << obj->_vertices[i] << std::endl;
	}
	obj->_cx += trans_vec[0];
	obj->_cy += trans_vec[1];
	obj->_cz += trans_vec[2];
	obj->init_x = obj->_cx;
	obj->init_y = obj->_cy;
	obj->init_z = obj->_cz;
}

void Scene::debug() {
	FILE* fp = fopen("./data/plane.obj_first_data.txt", "r");
	int vnum, band;
	fscanf(fp, "%d%d", &vnum, &band);
	int band2 = band * band;
	float* coef = new float[band2];
	for (int i = 0; i < 2000; ++i)for (int j = 0; j < band2; ++j)fscanf(fp, "%f", &coef[j]);
	for (int i = 0; i < band2; ++i)fscanf(fp, "%f", &coef[i]);
	//SphericalH::testVisMap(band, 96, coef, "./first_map.jpg");

	for (int i = 0; i < 8; ++i) {
		std::cout << "./data/plane.obj_second_" + name_list[i] + ".txt" << std::endl;
		FILE* now_file = fopen(("./data/plane.obj_second_" + name_list[i] + ".txt").c_str(), "r");
		fscanf(now_file, "%d%d", &vnum, &band);
		for (int i = 0; i < 2000; ++i)for (int j = 0; j < band2; ++j)fscanf(now_file, "%f", &coef[j]);
		for (int i = 0; i < band2; ++i)fscanf(now_file, "%f", &coef[i]);
		//SphericalH::testVisMap(band, 96, coef, "./second_map_"+std::to_string(i)+".jpg");
		fclose(now_file);
	}
	
	delete[] coef;
	fclose(fp);
}
