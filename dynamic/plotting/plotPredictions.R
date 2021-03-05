library(dplyr)
library(ggplot2)
library(tools) # for file_path_sans_ext()


plotPredictions <- function(actual_prediction_file_path) {
  df <- read.csv(actual_prediction_file_path)

  p <- df %>% ggplot(aes(x = EF_actual, y = EF_prediction)) + 
    geom_point(alpha = 0.8) +
    geom_smooth(method = 'lm', se = FALSE) + 
    ggpubr::stat_cor(method = 'pearson') +
    xlim(0,100) +
    ylim(0,100) +
    #scale_colour_viridis_c()
    geom_abline(slope=1, intercept=0) + 
    theme(text=element_text(size=24))
  
  # Save plot
  out_dir = dirname(actual_prediction_file_path)
  out_name_without_extension = file_path_sans_ext(basename(actual_prediction_file_path))
  out_name = paste0(out_name_without_extension, '.png')
  out_path = file.path(out_dir, out_name)
  
  aspect_ratio <- 1
  height <- 8
  width <- height*aspect_ratio
  
  ggsave(out_path, height=height, width=width)
}

file_path <- '/Volumes/ccmbio/sufkes/echonet/dynamic/output/video/r2plus1d_18_32_2_pretrained/train_EF_pred_and_actual.csv'
file_path <- '/Volumes/ccmbio/sufkes/echonet/dynamic/output/video/r2plus1d_18_32_2_pretrained/val_EF_pred_and_actual.csv'
#file_path <- '/Volumes/ccmbio/sufkes/echonet/dynamic/output-sickkids_frames1and2_tracings0/video/r2plus1d_18_32_2_pretrained/EF_pred_and_actual.csv'
#file_path <- '/Volumes/ccmbio/sufkes/echonet/dynamic/output-stanford-frames1and2_tracings0/video/r2plus1d_18_32_2_pretrained/EF_pred_and_actual.csv'
plotPredictions(file_path)

