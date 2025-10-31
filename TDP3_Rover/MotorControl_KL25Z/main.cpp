/*
 * FRDM-KL25Z — Dual Motor Control via L298 (MBED)
 * ------------------------------------------------
 * Explicit GPIO logic with adjustable PWM duty and timing.
 *
 * Motor driver: L298N
 * - Left motor: IN1, IN2, ENA (PWM)
 * - Right motor: IN3, IN4, ENB (PWM)
 * PWM frequency: 20 kHz
 */

#include "mbed.h"

// === Pin assignments (adjust if needed) ===
// Left motor
DigitalOut IN1(D7);
DigitalOut IN2(D6);
PwmOut     ENA(D5);

// Right motor
DigitalOut IN3(D4);
DigitalOut IN4(D3);
PwmOut     ENB(D2);

// RGB LEDs (onboard KL25Z)
DigitalOut LED_R(LED_RED);
DigitalOut LED_G(LED_GREEN);
DigitalOut LED_B(LED_BLUE);

// === Constants ===
#define PWM_FREQ_HZ 20000.0f
#define DUTY_MAX    1.0f   // mbed::PwmOut uses 0.0–1.0 range for duty cycle

// === LED helper ===
void leds_set(bool r, bool g, bool b) {
    LED_R = !r; // KL25Z LEDs are active-low
    LED_G = !g;
    LED_B = !b;
}

// === PWM helper ===
void motors_set_duty_sync(float left_duty, float right_duty) {
    ENA.write(left_duty);
    ENB.write(right_duty);
}

// === Safe shutdown ===
void motors_all_off() {
    motors_set_duty_sync(0.0f, 0.0f);

    // Coast both motors
    IN1 = 0; IN2 = 0;
    IN3 = 0; IN4 = 0;

    leds_set(false, false, false);
}

// === Motion routines ===
void move_forward(float duty, int duration_ms) {
    printf("Forward at %.0f%%\n", duty * 100);
    leds_set(false, true, false); // Green

    IN1 = 1; IN2 = 0; // Left forward
    IN3 = 1; IN4 = 0; // Right forward

    motors_set_duty_sync(duty, duty);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void move_backward(float duty, int duration_ms) {
    printf("Backward at %.0f%%\n", duty * 100);
    leds_set(false, false, true); // Blue

    IN1 = 0; IN2 = 1; // Left backward
    IN3 = 0; IN4 = 1; // Right backward

    motors_set_duty_sync(duty, duty);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_left_skid_reverse_inner(float duty_outer, int duration_ms) {
    printf("Turn left (reverse inner)\n");
    leds_set(true, false, true); // Magenta

    IN1 = 0; IN2 = 1; // Left reverse
    IN3 = 1; IN4 = 0; // Right forward

    motors_set_duty_sync(duty_outer, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_left_skid_coast_inner(float duty_outer, int duration_ms) {
    printf("Turn left (coast inner)\n");
    leds_set(true, true, false); // Yellow

    IN1 = 0; IN2 = 0; // Left coast
    IN3 = 1; IN4 = 0; // Right forward

    motors_set_duty_sync(0.0f, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_right_skid_reverse_inner(float duty_outer, int duration_ms) {
    printf("Turn right (reverse inner)\n");
    leds_set(false, true, true); // Cyan

    IN1 = 1; IN2 = 0; // Left forward
    IN3 = 0; IN4 = 1; // Right reverse

    motors_set_duty_sync(duty_outer, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_right_skid_coast_inner(float duty_outer, int duration_ms) {
    printf("Turn right (coast inner)\n");
    leds_set(true, false, false); // Red

    IN1 = 1; IN2 = 0; // Left forward
    IN3 = 0; IN4 = 0; // Right coast

    motors_set_duty_sync(duty_outer, 0.0f);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void motors_coast(int duration_ms) {
    printf("Coast\n");
    leds_set(true, false, false);
    motors_all_off();
    thread_sleep_for(duration_ms);
}

// === Main ===
int main() {
    printf("Dual-motor control (explicit GPIO logic, MBED, KL25Z)\n");

    // Setup PWM frequency
    ENA.period(1.0f / PWM_FREQ_HZ);
    ENB.period(1.0f / PWM_FREQ_HZ);

    // Ensure everything is off initially
    motors_all_off();

    while (true) {
        motors_coast(2000);

        move_forward(0.2f, 2000);
        move_forward(0.4f, 2000);
        move_forward(0.6f, 2000);
        move_forward(0.8f, 4000);
        turn_left_skid_reverse_inner(0.6f, 3000);
        turn_left_skid_coast_inner(0.6f, 3000);
        turn_right_skid_reverse_inner(0.6f, 3000);
        turn_right_skid_coast_inner(0.6f, 3000);
        move_backward(0.7f, 4000);
    }
}
