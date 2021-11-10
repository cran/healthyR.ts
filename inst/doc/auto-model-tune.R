## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(healthyR.ts)

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(timetk))
suppressPackageStartupMessages(library(tidymodels))
suppressPackageStartupMessages(library(modeltime))
suppressPackageStartupMessages(library(parallel))

## ----main_func, eval=FALSE----------------------------------------------------
#  ts_model_auto_tune(
#    .modeltime_model_id,
#    .calibration_tbl,
#    .splits_obj,
#    .drop_training_na = TRUE,
#    .date_col,
#    .value_col,
#    .tscv_assess = "12 months",
#    .tscv_skip = "6 months",
#    .slice_limit = 6,
#    .facet_ncol = 2,
#    .grid_size = 30,
#    .num_cores = 1,
#    .best_metric = "rmse"
#  )

## ----tune_template------------------------------------------------------------
ts_model_spec_tune_template("prophet")

## ----pxgb---------------------------------------------------------------------
ts_model_spec_tune_template("prophet_xgboost")

## ----data---------------------------------------------------------------------
data_ts <- AirPassengers

data_ts

## ----ts_to_tbl----------------------------------------------------------------
data_tbl <- ts_to_tbl(data_ts) %>%
  select(-index)

data_tbl

## ----ts_split-----------------------------------------------------------------
splits <- time_series_split(
    data_tbl
    , date_col
    , assess = 12
    , skip = 3
    , cumulative = TRUE
)

splits

head(training(splits))
head(testing(splits))

## ----rec_objs-----------------------------------------------------------------
rec_objs <- ts_auto_recipe(
  .data       = data_tbl
  , .date_col = date_col
  , .pred_col = value
)

rec_objs[[4]]

## ----wfsets-------------------------------------------------------------------
wfsets <- healthyR.ts::ts_wfs_mars(
  .model_type = "earth"
  , .recipe_list = rec_objs
)

wfsets

## ----wf_fits------------------------------------------------------------------
wf_fits <- wfsets %>%
  modeltime_fit_workflowset(
    data = training(splits)
    , control = control_fit_workflowset(
     allow_par = FALSE
     , verbose = TRUE
    )
  )

models_tbl <- wf_fits %>%
  filter(.model != "NULL")


models_tbl

## ----calibration_tbl----------------------------------------------------------
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

## ----tune_the_model-----------------------------------------------------------
output <- healthyR.ts::ts_model_auto_tune(
  .modeltime_model_id = 1,
  .calibration_tbl    = calibration_tbl,
  .splits_obj         = splits,
  .drop_training_na   = TRUE,
  .date_col           = date_col,
  .value_col          = value,
  .tscv_assess        = "12 months",
  .tscv_skip          = "3 months",
  .num_cores          = 1
)

## ----output_data--------------------------------------------------------------
output$data$calibration_tbl

output$data$calibration_tuned_tbl

output$data$tscv_data_tbl

output$data$tuned_results

output$data$best_tuned_results

output$data$tscv_obj

## ----output_model_info--------------------------------------------------------
output$model_info$model_spec

output$model_info$model_spec_engine

output$model_info$model_spec_tuner

output$model_info$plucked_model

output$model_info$wflw_tune_spec

output$model_info$grid_spec

output$model_info$tuned_tscv_wflw_spec

## ----output_plots, message=FALSE, warning=FALSE-------------------------------
output$plots$tune_results_plt

output$plots$tscv_plt

