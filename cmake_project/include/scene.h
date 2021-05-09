#pragma once
#include "diffuseObject.h"
#include "generalObject.h"
class Scene{
public:
	int obj_num;
	std::vector<std::string>name_list;
	std::vector<int>type_list;
	std::vector<Object*>obj_list;
	std::vector<glm::vec3>center;
	std::vector<glm::vec3>scale;
	std::vector<glm::vec3>color;
	int change_cnt;
	int _band;
	int vertice_num;
	int indices_num;

	void init(std::string& path);
	void prepare();
	void prepareData(int cur_band, int target_vertex = -1);
	bool change(glm::vec3& c_pos, glm::vec3& c_dir);
	void translate(Object* obj, glm::vec3 trans_vec);
	void debug();

};

