#include <stdio.h>

#define MAX_INPUT_LENGTH 5


void print_char_array(char s[]) {
    printf("%s\n", s);
}

void detab(int tab_length) {
    int c;
    char result[MAX_INPUT_LENGTH];

    int i = 0;

    while ((c = getchar()) != EOF) {
        // if (c != '\n')
        //     printf("%c\n", c);
        // fflush(stdout);

        if (c == '\t') {
            int tab_i = i % tab_length;
            int n_spaces = tab_length - tab_i;
            int j;
            for (j = 0; j < n_spaces; ++j) {
                result[i] = ' ';
                ++i;
            }
        } else {
            result[i] = c;
            ++i;
        }
    }
    result[i] = '\0';

    print_char_array(result);
}

int main() {
    int tab_length = 4;
    detab(tab_length);
}