#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <vector>
#include <string>
#include <algorithm>

#include "imgui/imgui.h"
#include "needaimbot.h"
#include "overlay.h"

// Local wrapped tooltip function for prediction UI
static void SetWrappedTooltip(const char* text)
{
    ImGui::BeginTooltip();
    ImVec2 window_size = ImGui::GetIO().DisplaySize;
    float max_width = window_size.x * 0.5f;
    ImGui::PushTextWrapPos(max_width);
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void draw_prediction()
{
    ImGui::Spacing();
    ImGui::SeparatorText("Prediction Settings");
    ImGui::Indent(10.0f);

    // Prediction algorithm selection
    const char* prediction_algorithms[] = {
        "None",
        "Velocity Based",
        "Linear Regression",
        "Exponential Smoothing",
        "Kalman Filter"
    };
    int current_algorithm_index = 0;
    std::string current_algo = config.prediction_algorithm;
    if (current_algo == "Velocity Based")          { current_algorithm_index = 1; }
    else if (current_algo == "Linear Regression")  { current_algorithm_index = 2; }
    else if (current_algo == "Exponential Smoothing") { current_algorithm_index = 3; }
    else if (current_algo == "Kalman Filter")      { current_algorithm_index = 4; }

    ImGui::PushItemWidth(150);
    if (ImGui::Combo("Algorithm", &current_algorithm_index, prediction_algorithms, IM_ARRAYSIZE(prediction_algorithms)))
    {
        config.prediction_algorithm = prediction_algorithms[current_algorithm_index];
        config.saveConfig();
        prediction_settings_changed.store(true);
    }
    ImGui::PopItemWidth();
    if (ImGui::IsItemHovered())
    {
        SetWrappedTooltip(
            "Select the prediction method:\n"
            "None: No prediction.\n"
            "Velocity Based: Simple prediction based on current velocity.\n"
            "Linear Regression: Predicts based on linear fit of past movement.\n"
            "Exponential Smoothing: Weighted average prediction, favors recent data.\n"
            "Kalman Filter: Statistical filter for noisy data (requires tuning)."
        );
    }
    ImGui::Spacing();

    // Algorithm-specific settings
    if (config.prediction_algorithm == "Velocity Based")
    {
        ImGui::SeparatorText("Velocity Prediction Settings");
        ImGui::PushItemWidth(100);
        if (ImGui::InputFloat("Prediction Time (ms)", &config.velocity_prediction_ms, 1.0f, 5.0f, "%.1f"))
        {
            config.velocity_prediction_ms = std::max(0.0f, config.velocity_prediction_ms);
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("How far ahead in milliseconds to predict the target's position based on its current velocity.");
        }
    }
    else if (config.prediction_algorithm == "Linear Regression")
    {
        ImGui::SeparatorText("Linear Regression Settings");
        ImGui::PushItemWidth(100);
        ImGui::Text("Past Points (N):"); ImGui::SameLine();
        if (ImGui::InputInt("##LRPastPoints", &config.lr_past_points, 1, 5))
        {
            config.lr_past_points = std::max(2, config.lr_past_points);
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Number of past target positions to use for calculating the regression line. More points = smoother but less responsive.");
        }
    }
    else if (config.prediction_algorithm == "Exponential Smoothing")
    {
        ImGui::SeparatorText("Exponential Smoothing Settings");
        ImGui::PushItemWidth(100);
        ImGui::Text("Factor (Alpha):"); ImGui::SameLine();
        if (ImGui::SliderFloat("##ESAlpha", &config.es_alpha, 0.01f, 1.0f, "%.2f"))
        {
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Smoothing factor between 0.01 and 1.0. Controls weighting of recent vs. past data.");
        }

        ImGui::PushItemWidth(100);
        ImGui::Text("Trend Factor (Beta):"); ImGui::SameLine();
        if (ImGui::SliderFloat("##ESBeta", &config.es_beta, 0.01f, 1.0f, "%.2f"))
        {
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        ImGui::PopItemWidth();
        if (ImGui::IsItemHovered())
        {
            SetWrappedTooltip("Trend smoothing factor between 0.01 and 1.0. Controls smoothing of velocity trend.");
        }
    }
    else if (config.prediction_algorithm == "Kalman Filter")
    {
        ImGui::SeparatorText("Kalman Filter Settings");
        ImGui::PushItemWidth(100);
        ImGui::Text("Process Noise (Q):"); ImGui::SameLine();
        float temp_q = static_cast<float>(config.kalman_q);
        if (ImGui::InputFloat("##KalmanQ", &temp_q, 0.001f, 0.01f, "%.3f"))
        {
            config.kalman_q = static_cast<double>(temp_q);
            if (config.kalman_q < 0) config.kalman_q = 0;
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        if (ImGui::IsItemHovered()) SetWrappedTooltip("Represents the uncertainty in the target's movement model.");

        ImGui::Text("Measurement Noise (R):"); ImGui::SameLine();
        float temp_r = static_cast<float>(config.kalman_r);
        if (ImGui::InputFloat("##KalmanR", &temp_r, 0.001f, 0.01f, "%.3f"))
        {
            config.kalman_r = static_cast<double>(temp_r);
            if (config.kalman_r < 0) config.kalman_r = 0;
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        if (ImGui::IsItemHovered()) SetWrappedTooltip("Represents the uncertainty in the measurements.");

        ImGui::Text("Estimate Error (P):"); ImGui::SameLine();
        float temp_p = static_cast<float>(config.kalman_p);
        if (ImGui::InputFloat("##KalmanP", &temp_p, 0.001f, 0.01f, "%.3f"))
        {
            config.kalman_p = static_cast<double>(temp_p);
            if (config.kalman_p < 0) config.kalman_p = 0;
            config.saveConfig();
            prediction_settings_changed.store(true);
        }
        if (ImGui::IsItemHovered()) SetWrappedTooltip("Initial estimate of the state covariance.");
        ImGui::PopItemWidth();
    }

    ImGui::Unindent(10.0f);
    ImGui::Spacing();
} 