#ifndef CLI_H
#define CLI_H

typedef struct cli_arguments {
    int isError;
    int showGpus; // -g --gpu
    int device; // -d=0 --device=0
    int train; // -t
    int predict;  // -p
    char loadPath[1000]; // -l= -l xxx --load
    char savePath[1000]; // -s
    char configPath[1000]; // -c
} cli_arguments_t;

int parseCliArguments(cli_arguments_t* cli, int argc, const char* argv[]);

#endif // CLI_H
