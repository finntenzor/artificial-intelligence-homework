/**
 * config.cpp
 */

#include <cstdio>
#include <cstring>

#include "config.h"

Config::Config(void* dist) {
    this->dist = dist;
    beforeModuleCallback = NULL;
}

Config::~Config() {
    dist = NULL;
    beforeModuleCallback = NULL;
}

void Config::expectString(const char* module, const char* field, char* value) {
    items.push_back({
        module,
        field,
        CONFIG_TYPE_STRING,
        (void*)value
    });
}

void Config::expectInteger(const char* module, const char* field, int* value) {
    items.push_back({
        module,
        field,
        CONFIG_TYPE_INTEGER,
        (void*)value
    });
}

void Config::expectDouble(const char* module, const char* field, double* value) {
    items.push_back({
        module,
        field,
        CONFIG_TYPE_DOUBLE,
        (void*)value
    });
}

void Config::expectLayer(const char* module, const char* field, config_layer_callback_t value) {
    items.push_back({
        module,
        field,
        CONFIG_TYPE_LAYER,
        (void*)value
    });
}

void Config::beforeModule(config_before_module_callback_t callback) {
    beforeModuleCallback = callback;
}

int Config::readItem(FILE* f, const char* module, const char* field) {
    int numbers[100];
    int k;
    int n = items.size();
    for (int i = 0; i < n; i++) {
        config_item_t* item = &items[i];
        if ((strcmp(item->module, module) == 0) && (strcmp(item->field, field) == 0)) {
            switch (item->type) {
            case CONFIG_TYPE_STRING:
                fscanf(f, "%s", (char*)(item->value));
                break;
            case CONFIG_TYPE_INTEGER:
                fscanf(f, "%d", (int*)(item->value));
                break;
            case CONFIG_TYPE_DOUBLE:
                fscanf(f, "%lf", (double*)(item->value));
                break;
            case CONFIG_TYPE_LAYER: {
                    config_layer_callback_t callback = (config_layer_callback_t)(item->value);
                    int ret = 0;
                    k = 0;
                    while (fscanf(f, "%d", &numbers[k]) == 1) {
                        k++;
                    }
                    ret = (*callback)(dist, field, k, numbers);
                    if (ret) return ret;
                }
                break;
            default:
                fprintf(stderr, "未知类型: %d", item->type);
                break;
            }
            return 0;
        }
    }
    return 1;
}

int Config::read(const char* filepath) {
    int ret = 0;
    FILE* f;
    char module[100];
    char temp[1000];
    int sl;

    f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "无法打开文件 %s\n", filepath);
        return -1;
    }

    while (!feof(f)) {
        fscanf(f, "%s", temp);
        sl = strlen(temp);
        if (temp[0] == '[' && temp[sl - 1] == ']') {
            if (sl >= 100) {
                fprintf(stderr, "模块名太长,配置文件是否写错了?\n");
                ret = 1;
                break;
            }
            temp[sl - 1] = '\0';
            strcpy(module, &temp[1]);
            if (beforeModuleCallback) {
                ret = (*beforeModuleCallback)(dist, module);
                if (ret) break;
            }
        } else {
            if (readItem(f, module, temp)) {
                fprintf(stderr, "没有找到该配置: [%s].%s\n", module, temp);
                ret = 1;
                break;
            }
        }
    }

    fclose(f);
    return ret;
}
