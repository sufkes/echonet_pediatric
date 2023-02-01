#!/usr/bin/env Rscript
library(argparser)
suppressPackageStartupMessages(library(dplyr))
library(ggplot2)
library(tools) # for file_path_sans_ext()

plotPredictions <- function(file_path, out_dir, axis_min, axis_max, suffix, size_multiplier, hide_y_axis) {
  df <- read.csv(file_path)

  # Rename actual and prediction columns to appease ggplot.
  is_lvot = "lvot_actual" %in% colnames(df)
  is_vti = "vti_actual" %in% colnames(df)
  is_co = "co_actual" %in% colnames(df)
  is_ci = "ci_actual" %in% colnames(df)
  for (col in colnames(df)) {
      if (grepl('actual$', col)) {
        df['actual'] <- df[col]
      }
      else if (grepl('prediction$', col)) {
        df['prediction'] <- df[col]
      }
  }
  
  min_val=min(min(df['actual']), min(df['prediction']))
  max_val=max(max(df['actual']), max(df['prediction']))
  if (is.na(axis_min)) {
    axis_min = min_val - 0.05*(max_val - min_val)
  }
  if (is.na(axis_max)) {
    axis_max = max_val + 0.05*(max_val - min_val)
  }
  
  if (!is.na(suffix)) {
    y_label <- paste0("predicted ", suffix)
    x_label <- paste0("actual ", suffix)
  } else if (is_lvot) {
    y_label <- bquote('predicted (mm)') # for Cardiac Output(CO)
    x_label <- bquote('actual (mm)')   
  } else if (is_vti) {
    y_label <- bquote('predicted (cm)') # for Cardiac Output(CO)
    x_label <- bquote('actual (cm)')
  } else if (is_co) {
    y_label <- bquote('predicted (L/min)') # for Cardiac Output(CO)
    x_label <- bquote('actual (L/min)')
  } else if (is_ci) {
    y_label <- bquote('predicted (L/min/'*m^2*')') # for Cardiac Index (CI)
    x_label <- bquote('actual (L/min/'*m^2*')')
  } else {
    y_label <- "predicted"
    x_label <- "actual"
  }

  axis_text_size <- 28*size_multiplier
  text_size <- 32*size_multiplier
  point_size <- 2.2*size_multiplier
  
  p <- df %>% ggplot(aes(x=actual, y=prediction)) + 
    geom_point(alpha = 1) +
    #geom_smooth(method = 'lm', se = FALSE) + 
    #ggpubr::stat_cor(method = 'pearson') +
    #xlim(axis_min, axis_max) +
    #ylim(axis_min, axis_max) +
    #coord_cartesian(xlim = c(axis_min, axis_max), ylim = c(axis_min, axis_max), expand = c(0, 0)) +
    #coord_cartesian(xlim = c(axis_min, axis_max), ylim = c(axis_min, axis_max)) +
    coord_fixed(ratio = 1, xlim = c(axis_min, axis_max), ylim = c(axis_min, axis_max), expand = FALSE, clip = "on") + 
    labs(y=y_label, x=x_label) +
    #scale_colour_viridis_c()
    geom_abline(slope=1, intercept=0) + 
    geom_point(size=point_size, color='blue') +
    theme(text=element_text(size=text_size, color='black'),
          axis.text=element_text(size=axis_text_size, color='black'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_line(colour = "black")) + 
#    {if(hide_y_axis==TRUE)theme(axis.title.y=element_blank(),
#                                axis.text.y=element_blank())}
  {if(hide_y_axis==TRUE)theme(axis.title.y=element_text(size=text_size, color='white'),
                              axis.text.y=element_text(size=axis_text_size, color='white'))}
  
  # Save plot
  if (is.na(out_dir)) {
    out_dir = dirname(file_path)
  }
  out_name_without_extension = file_path_sans_ext(basename(file_path))
  out_name = paste0(out_name_without_extension, '.png')
  out_path = file.path(out_dir, out_name)
  
  aspect_ratio <- 1
  height <- 8
  width <- height*aspect_ratio
  
  ggsave(out_path, height=height, width=width)
}

description <- 'Plot predicted versus actual.'
parser <- arg_parser(description)
parser <- add_argument(parser, 'file_path', help='path to CSV file with actual and predicted values')
parser <- add_argument(parser, '--out_dir', short='-o', help='output directory. Default: save to same directory as input CSV file.', default=NULL)
parser <- add_argument(parser, '--axis_min', short='-l', help='axis min', type="numeric")
parser <- add_argument(parser, '--axis_max', short='-u', help='axis max', type="numeric")
parser <- add_argument(parser, '--suffix', short='-s', help='suffix for axis labels (e.g. "(cm)")', type="character")
parser <- add_argument(parser, '--size_multiplier', short='-z', help='increase font and point sizes by this factor', type="numeric", default=1)
parser <- add_argument(parser, '--hide_y_axis', help='hide y axis label', type="logical", default=FALSE)
args <- parse_args(parser)

#plotPredictions(file_path)
#print('Args raw.')
#print(args) # first three things in args are junk; get rid of them?

args[[1]] <- NULL
args[['help']] <- NULL
args[['opts']] <- NULL

#print(args)
#print('Args with junk removed.')
#print(args) # first three things in args are junk; get rid of them?

do.call(plotPredictions, args)