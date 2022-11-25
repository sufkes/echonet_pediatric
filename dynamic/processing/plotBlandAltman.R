#!/usr/bin/env Rscript
library(argparser)
library(dplyr)
library(ggplot2)
library(blandr)
library(tools) # for file_path_sans_ext()

plotPredictions <- function(file_path, out_dir, label_pos_x_rel=0.2, title=NA, auto_bounds=FALSE) {
  df <- read.csv(file_path)

  # Rename actual and prediction columns to appease ggplot.
  for (col in colnames(df)) {
      if (grepl('actual$', col)) {
        df['actual'] <- df[col]
      }
      else if (grepl('prediction$', col)) {
        df['prediction'] <- df[col]
      }
  }

  if (auto_bounds == TRUE) {
    min_val=min(min(df['actual']), min(df['prediction']))
    max_val=max(max(df['actual']), max(df['prediction']))
    axis_min = min_val - 0.05*(max_val - min_val)
    axis_max = max_val + 0.05*(max_val - min_val)
  } else {
    axis_min=0
    axis_max=80
  }
  df[,'difference'] = df['prediction'] - df['actual']
  y_min = min(df[['difference']])
  y_max = max(df[['difference']])
  avg = mean(df[['difference']])

  # Manually set the x position of the error bars.
  #error_bar_x_rel = 0.20 # relative position along x-axis for the error bars to be placed.
  error_bar_x = axis_min + (axis_max - axis_min) * label_pos_x_rel
  label_x_offset_rel = 0.006
  label_x_offset = (axis_max-axis_min) * label_x_offset_rel
  label_y_offset_rel = 0.04
  label_y_offset = (y_max - y_min) * label_y_offset_rel
  if (avg<0) {
     label_y_offset <- -label_y_offset
  }
  
  ## Get stats from Bland-Altman analysis
  blstats <- blandr.statistics(df$prediction, df$actual, sig.level=0.95) # sig.level is for confidence intervals; limit of agreement is always 95%
  #print(blstats)
  
  ## Create Bland-Altman plot
  # Set axis labels.
  y_lab = "predicted - actual"
  x_lab = "mean of predicted and actual"

  p <- blandr.draw(df$prediction, df$actual, ciDisplay=FALSE, ciShading=FALSE, plotTitle=NULL) +
    theme(text=element_text(size=16, color='black'),
          axis.text=element_text(size=14, color='black'),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_rect(colour = "black", fill=NA, size=1)) +
    
    # Set aspect ratio
    coord_fixed(ratio=1.5) +
     
    labs(y=y_lab, x=x_lab) +

    # 95% CI for bias
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$biasLowerCI, ymax=blstats$biasUpperCI), width=1) + 
    annotate("text", x=error_bar_x+label_x_offset, y=blstats$bias + label_y_offset, label=paste("mean difference =", sprintf("%.2f",blstats$bias)), hjust=0, size=4) +
    
    # 95% CI for upper limit of agreement
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$upperLOA_lowerCI, ymax=blstats$upperLOA_upperCI)) +
    annotate("text", x=error_bar_x+label_x_offset, y=blstats$upperLOA + label_y_offset, label=paste("mean difference + 1.96SD =", sprintf("%.2f",blstats$upperLOA)), hjust=0, size=4) +
    
    # 95% CI for lower limit of agreement
    geom_errorbar(mapping=aes(x=error_bar_x, ymin=blstats$lowerLOA_lowerCI, ymax=blstats$lowerLOA_upperCI)) +
    annotate("text", x=error_bar_x+label_x_offset, y=blstats$lowerLOA + label_y_offset, label=paste("mean difference - 1.96SD =", sprintf("%.2f",blstats$lowerLOA)), hjust=0, size=4)

  # Set title if one was specified.
  if ( !is.na(title)) {
     p <- p + ggtitle(title)
  }
    
  # Save plot
  if (is.na(out_dir)) {
    out_dir = dirname(file_path)
  }
  out_name_without_extension = file_path_sans_ext(basename(file_path))
  out_name = paste0(out_name_without_extension, '-BlandAltman.png')
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
parser <- add_argument(parser, '--auto_bounds', short='-a', help='automatically determine axis bounds for figures.', flag=TRUE)
parser <- add_argument(parser, '--label_pos_x_rel', short='-e', help='relative position of error bars along the x axis.', default=0.20)
parser <- add_argument(parser, '--title', short='-t', help='title text', default=NA)
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