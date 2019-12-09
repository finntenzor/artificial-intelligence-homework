/**
 * cli.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cli.h"

static const int ARG_SHOW_GPUS = 1;
static const int ARG_DEVICE = 2;
static const int ARG_LOAD_PATH = 3;
static const int ARG_SAVE_PATH = 4;
static const int ARG_TRAIN = 5;
static const int ARG_PREDICT = 6;
static const int ARG_CONFIG = 7;
static const int ARG_VERSION = 8;

static const int STATUS_ERROR = -1;
static const int STATUS_EXIT = 0;
static const int STATUS_READY = 1; // ׼���õȴ���һ������
static const int STATUS_DASH = 2; // �Ѷ�ȡ��һ�����
static const int STATUS_SHORT_ARG = 3; // �Ѷ�ȡ��һ���ַ���ȷ���Ƕ̲���
static const int STATUS_CONFIRM_EXPECT = 4; // ��֪�����������ݲ�����ת���ڴ������������ڴ�����
static const int STAUTS_EXPECT_END = 5; // �ò����ڴ�������
static const int STATUS_EXPECT_INPUT = 6; // �ò����ڴ�����
static const int STATUS_READY_INLINE_INPUT = 7; // �Ѿ���ȡ���Ⱥţ��ȴ���������
static const int STATUS_READY_SPACE_INPUT = 8; // �Ѿ���ȡ���ո񣬵ȴ���������
static const int STATUS_DOUBLE_DASH = 9; // �Ѿ�����������ܣ�ȷ���ǳ�����

void clearCliArguments(cli_arguments_t* cli) {
    cli->isError = 0;
    cli->showGpus = 0;
    cli->device = 0;
    cli->train = 0;
    cli->predict = 0;
    cli->version = 0;
    strcpy(cli->loadPath, "");
    strcpy(cli->savePath, "");
    strcpy(cli->configPath, "");
}

int getArgFromLongArgument(const char* str, int* currentPos) {
    char argumentName[100];
    int arg = 0;
    int pos = *currentPos;
    sscanf(str, "%[a-zA-Z0-9]", argumentName);
    if (strcmp(argumentName, "gpu") == 0) {
        arg = ARG_SHOW_GPUS;
        pos += 3;
    } else if (strcmp(argumentName, "device") == 0) {
        arg = ARG_DEVICE;
        pos += 6;
    } else if (strcmp(argumentName, "load") == 0) {
        arg = ARG_LOAD_PATH;
        pos += 4;
    } else if (strcmp(argumentName, "save") == 0) {
        arg = ARG_SAVE_PATH;
        pos += 4;
    } else if (strcmp(argumentName, "train") == 0) {
        arg = ARG_TRAIN;
        pos += 5;
    } else if (strcmp(argumentName, "predict") == 0) {
        arg = ARG_PREDICT;
        pos += 7;
    } else if (strcmp(argumentName, "config") == 0) {
        arg = ARG_CONFIG;
        pos += 6;
    } else if (strcmp(argumentName, "version") == 0) {
        arg = ARG_VERSION;
        pos += 7;
    }
    *currentPos = pos;
    return arg;
}

int parseCliArguments(cli_arguments_t* cli, int argc, const char* argv[]) {
    int status = STATUS_READY;
    int index = 1;
    int currentArg = 0;
    int currentPos = 0;

    clearCliArguments(cli);

    while (!(status == STATUS_EXIT || status == STATUS_ERROR)) {
        const char* arg = NULL;

        if (index >= argc) {
            if (status == STATUS_READY) {
                status = STATUS_EXIT;
                break;
            } else {
                fprintf(stderr, "ѡ����� %s\n", argv[index - 1]);
                status = STATUS_ERROR;
                break;
            }
        }
        arg = argv[index];

        switch (status) {
            case STATUS_READY:
                if (arg[currentPos] == '-') {
                    status = STATUS_DASH;
                    currentPos++;
                } else {
                    fprintf(stderr, "�޷�ʶ���ѡ�� %s\n", arg);
                    status = STATUS_ERROR;
                }
                break;
            case STATUS_DASH:
                if (arg[currentPos] == '-') {
                    status = STATUS_DOUBLE_DASH;
                    currentPos++;
                } else if (arg[currentPos] >= 'a' && arg[currentPos] <= 'z') {
                    status = STATUS_SHORT_ARG;
                } else {
                    fprintf(stderr, "�޷�ʶ���ѡ�� %s\n", arg);
                    status = STATUS_ERROR;
                }
                break;
            case STATUS_SHORT_ARG:
                if (arg[currentPos] == 'g') {
                    currentArg = ARG_SHOW_GPUS;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 'd') {
                    currentArg = ARG_DEVICE;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 'l') {
                    currentArg = ARG_LOAD_PATH;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 's') {
                    currentArg = ARG_SAVE_PATH;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 't') {
                    currentArg = ARG_TRAIN;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 'p') {
                    currentArg = ARG_PREDICT;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 'c') {
                    currentArg = ARG_CONFIG;
                    status = STATUS_CONFIRM_EXPECT;
                } else if (arg[currentPos] == 'v') {
                    currentArg = ARG_VERSION;
                    status = STATUS_CONFIRM_EXPECT;
                } else {
                    fprintf(stderr, "�޷�ʶ���ѡ�� %s\n", arg);
                    status = STATUS_ERROR;
                }
                currentPos++;
                break;
            case STATUS_CONFIRM_EXPECT:
                if (currentArg == ARG_SHOW_GPUS) {
                    status = STAUTS_EXPECT_END;
                } else if (currentArg == ARG_DEVICE) {
                    status = STATUS_EXPECT_INPUT;
                } else if (currentArg == ARG_LOAD_PATH) {
                    status = STATUS_EXPECT_INPUT;
                } else if (currentArg == ARG_SAVE_PATH) {
                    status = STATUS_EXPECT_INPUT;
                } else if (currentArg == ARG_TRAIN) {
                    status = STAUTS_EXPECT_END;
                } else if (currentArg == ARG_PREDICT) {
                    status = STAUTS_EXPECT_END;
                } else if (currentArg == ARG_CONFIG) {
                    status = STATUS_EXPECT_INPUT;
                } else if (currentArg == ARG_VERSION) {
                    status = STAUTS_EXPECT_END;
                }
                break;
            case STAUTS_EXPECT_END:
                if (arg[currentPos] == '\0') {
                    if (currentArg == ARG_SHOW_GPUS) {
                        cli->showGpus = 1;
                        index++;
                        currentPos = 0;
                        status = STATUS_READY;
                    } else if (currentArg == ARG_TRAIN) {
                        cli->train = 1;
                        index++;
                        currentPos = 0;
                        status = STATUS_READY;
                    } else if (currentArg == ARG_PREDICT) {
                        cli->predict = 1;
                        index++;
                        currentPos = 0;
                        status = STATUS_READY;
                    } else if (currentArg == ARG_VERSION) {
                        cli->version = 1;
                        index++;
                        currentPos = 0;
                        status = STATUS_READY;
                    } else {
                        fprintf(stderr, "���ɴ���� EXPECT END\n");
                        status = STATUS_ERROR;
                    }
                } else {
                    fprintf(stderr, "ѡ�� %s ����������\n", arg);
                    status = STATUS_ERROR;
                }
                break;
            case STATUS_EXPECT_INPUT:
                if (arg[currentPos] == '=') {
                    status = STATUS_READY_INLINE_INPUT;
                    currentPos++;
                } else if (arg[currentPos] == '\0') {
                    status = STATUS_READY_SPACE_INPUT;
                    index++;
                    currentPos = 0;
                } else {
                    fprintf(stderr, "ѡ�� %s Ҫ������\n", arg);
                    status = STATUS_ERROR;
                }
                break;
            case STATUS_READY_INLINE_INPUT:
            case STATUS_READY_SPACE_INPUT:
                if (currentArg == ARG_DEVICE) {
                    if (sscanf(arg + currentPos, "%d", &cli->device) == 1) {
                        status = STATUS_READY;
                    } else {
                        fprintf(stderr, "�޷���ȡѡ�� %s ������\n", arg);
                    }
                } else if (currentArg == ARG_LOAD_PATH) {
                    strcpy(cli->loadPath, arg + currentPos);
                    status = STATUS_READY;
                } else if (currentArg == ARG_SAVE_PATH) {
                    strcpy(cli->savePath, arg + currentPos);
                    status = STATUS_READY;
                } else if (currentArg == ARG_CONFIG) {
                    strcpy(cli->configPath, arg + currentPos);
                    status = STATUS_READY;
                } else {
                    fprintf(stderr, "���ɴ���� READY INPUT\n");
                    status = STATUS_ERROR;
                }
                index++;
                currentPos = 0;
                break;
            case STATUS_DOUBLE_DASH:
                currentArg = getArgFromLongArgument(arg + currentPos, &currentPos);
                if (currentArg == 0) {
                    fprintf(stderr, "�޷�ʶ���ѡ�� %s\n", arg);
                    status = STATUS_ERROR;
                } else {
                    status = STATUS_CONFIRM_EXPECT;
                }
                break;
            default:
                break;
        }
    }
    if (status == STATUS_ERROR) {
        cli->isError = 1;
        fprintf(stderr, "�����н������߼�⵽���������쳣��Ϣ�Ѿ����\n");
    }
    return cli->isError;
}
