/*
 * FRDM-KL25Z — Dual Motor Control via L298 (MBED)
 * ------------------------------------------------
 * Explicit GPIO logic with adjustable PWM duty and timing.
 *
 * Motor driver: L298N
 * - Left motor: left_in1, left_in2, left_pwm (PWM)
 * - Right motor: right_in1, right_in2, right_pwm (PWM)
 * PWM frequency: 20 kHz
 */

#include "mbed.h"

// === Pin assignments (adjust if needed) ===
// Left motor
DigitalOut left_in1(D7);
DigitalOut left_in2(D6);
PwmOut     left_pwm(D5);

// Right motor
DigitalOut right_in1(D4);
DigitalOut right_in2(D3);
PwmOut     right_pwm(D2);

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
    left_pwm.write(left_duty);
    right_pwm.write(right_duty);
}

// === Safe shutdown ===
void motors_all_off() {
    motors_set_duty_sync(0.0f, 0.0f);

    // Coast both motors
    left_in1 = 0; left_in2 = 0;
    right_in1 = 0; right_in2 = 0;

    leds_set(false, false, false);
}

// === Motion routines ===
void move_forward(float duty, int duration_ms) {
    printf("Forward at %.0f%%\n", duty * 100);
    leds_set(false, true, false); // Green

    left_in1 = 1; left_in2 = 0; // Left forward
    right_in1 = 1; right_in2 = 0; // Right forward

    motors_set_duty_sync(duty, duty);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void move_backward(float duty, int duration_ms) {
    printf("Backward at %.0f%%\n", duty * 100);
    leds_set(true, false, false);

    left_in1 = 0; left_in2 = 1; // Left backward
    right_in1 = 0; right_in2 = 1; // Right backward

    motors_set_duty_sync(duty, duty);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_left_skid_reverse_inner(float duty_outer, int duration_ms) {
    printf("Turn left (reverse inner)\n");
    leds_set(true, false, true); // Magenta

    left_in1 = 0; left_in2 = 1; // Left reverse
    right_in1 = 1; right_in2 = 0; // Right forward

    motors_set_duty_sync(duty_outer, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_left_skid_coast_inner(float duty_outer, int duration_ms) {
    printf("Turn left (coast inner)\n");
    leds_set(true, true, false); // Yellow

    left_in1 = 0; left_in2 = 0; // Left coast
    right_in1 = 1; right_in2 = 0; // Right forward

    motors_set_duty_sync(0.0f, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_right_skid_reverse_inner(float duty_outer, int duration_ms) {
    printf("Turn right (reverse inner)\n");
    leds_set(false, true, true); // Cyan

    left_in1 = 1; left_in2 = 0; // Left forward
    right_in1 = 0; right_in2 = 1; // Right reverse

    motors_set_duty_sync(duty_outer, duty_outer);
    thread_sleep_for(duration_ms);
    motors_all_off();
}

void turn_right_skid_coast_inner(float duty_outer, int duration_ms) {
    printf("Turn right (coast inner)\n");
    leds_set(true, false, false); // Red

    left_in1 = 1; left_in2 = 0; // Left forward
    right_in1 = 0; right_in2 = 0; // Right coast

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
    left_pwm.period(1.0f / PWM_FREQ_HZ);
    right_pwm.period(1.0f / PWM_FREQ_HZ);

    // Ensure everything is off initially
    motors_all_off();

    while (true) {

        move_forward(0.1f, 2000);
        move_forward(0.2f, 2000);
        move_forward(0.3f, 2000);
        move_forward(0.4f, 4000);
        move_forward(0.5f, 2000);
        move_forward(0.6f, 2000);
        move_forward(0.7f, 2000);
        /*
        move_forward(0.9f, 4000);
        move_forward(1.0f, 2000);
        
        turn_left_skid_reverse_inner(0.6f, 3000);
        turn_left_skid_coast_inner(0.6f, 3000);
        turn_right_skid_reverse_inner(0.6f, 3000);
        turn_right_skid_coast_inner(0.6f, 3000);
        move_backward(0.7f, 4000);
        */
    }
}
