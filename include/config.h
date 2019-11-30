#ifndef CONFIG_H
#define CONFIG_H

#ifndef CHECK
#include <vector>
#include <cstdio>
#endif

const int CONFIG_TYPE_STRING = 1;
const int CONFIG_TYPE_INTEGER = 2;
const int CONFIG_TYPE_DOUBLE = 3;
const int CONFIG_TYPE_LAYER = 4;

typedef int (*config_layer_callback_t)(void* dist, const char* key, const int argc, const int argv[]);
typedef int (*config_before_module_callback_t)(void* dist, const char* module);

typedef struct config_item {
    const char* module;
    const char* field;
    int type;
    void* value;
} config_item_t;

class Config {
    void* dist;
    config_before_module_callback_t beforeModuleCallback;
    std::vector<config_item_t> items;
protected:
    int readItem(FILE* f, const char* module, const char* temp);
public:
    Config(void* dist);
    ~Config();
    void expectString(const char* module, const char* field, char* offset);
    void expectInteger(const char* module, const char* field, int* offset);
    void expectDouble(const char* module, const char* field, double* offset);
    void expectLayer(const char* module, const char* field, config_layer_callback_t offset);
    void beforeModule(config_before_module_callback_t callback);
    int read(const char* filepath);
};

#endif // CONFIG_H
