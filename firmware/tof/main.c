/* Range with a VL53L5CX or VL53L8CX and print each 8x8 frame as one line of 64 distances in mm (-1 = no valid
 * target in that zone), row by row in the sensor's own order.
 *
 *     tof_vl53l5cx [/dev/i2c-3] [Hz]
 *
 * Built once per generation (-DGEN=5 or 8) by build.sh, on the duck. ST's driver (vl53l?cx/) and the Linux platform
 * layer and flat shim around it (platform.c, probe.c, shim.c) come from pollen-robotics/microduck `tof/vendor`.
 * It never sits silent: if no frame comes for 3 s (at the start or later), it starts itself again with a full reset
 * of the sensor and a fresh firmware upload, which takes about 8 s. A sensor left half-configured by another program
 * otherwise makes ST's driver wait for minutes.
 * Exit codes: 1 bus, 2 no sensor answering, 3 firmware upload failed, 4 could not start, 5/6 bus error while ranging. */
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if GEN == 8
#define P(x) vl8_##x
#else
#define P(x) vl5_##x
#endif
int P(open)(const char *, uint8_t);
int P(is_alive)(void);
int P(init)(void);
int P(start)(uint8_t);
int P(data_ready)(void);
int P(get_frame)(int16_t *, uint8_t *);

static char **self, exe[512];

static void silent(int sig)      /* no frame in time: run again from the top, this time resetting the sensor first */
{
    char *again[] = { self[0], self[1], self[2], "reset", NULL };
    execv(exe, again);
    _exit(4);
}

int main(int argc, char **argv)
{
    const char *bus = argc > 1 ? argv[1] : "/dev/i2c-3";
    int hz = argc > 2 ? atoi(argv[2]) : 15;
    int reset = argc > 3;
    int16_t mm[64];
    uint8_t status[64];
    static char *args[3];
    args[0] = argv[0]; args[1] = (char *)bus; args[2] = argc > 2 ? argv[2] : "15";
    self = args;
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return 1;
    exe[n] = 0;
    signal(SIGALRM, silent);
    sigset_t alrm;                 /* a program started from the signal handler inherits SIGALRM blocked: unblock */
    sigemptyset(&alrm); sigaddset(&alrm, SIGALRM); sigprocmask(SIG_UNBLOCK, &alrm, NULL);
    if (P(open)(bus, 0x29)) { perror(bus); return 1; }
    if (!P(is_alive)()) { fprintf(stderr, "no sensor at 0x29 on %s\n", bus); return 2; }
    /* A sensor that already has its firmware (this program ranged with it since power-up) starts at once, and the
     * upload takes 8 s. So: first try to range as it is, and reset and upload only if no frame comes within 3 s. */
    alarm(reset ? 15 : 3);    /* the reset and upload take 8.4 s when they work */
    if (reset) {
        if (P(init)()) { fprintf(stderr, "firmware upload failed\n"); return 3; }
        alarm(3);
    }
    if (P(start)((uint8_t)hz)) { fprintf(stderr, "could not start ranging\n"); return 4; }
    for (;;) {
        int ready = P(data_ready)();
        if (ready < 0) return 5;
        if (!ready) { usleep(4000); continue; }
        if (P(get_frame)(mm, status)) return 6;
        alarm(3);
        for (int i = 0; i < 64; i++)       /* ST: status 5 and 9 are the valid ones */
            printf("%d%c", status[i] == 5 || status[i] == 9 ? mm[i] : -1, i == 63 ? '\n' : ' ');
        fflush(stdout);
    }
}
