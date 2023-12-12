# Developmental PreTraining (DPT) for Object Recognition

## Overview

This repository introduces Developmental PreTraining (DPT), a curriculum-based pre-training approach designed to address the growing data requirements of Deep Neural Networks (DNNs) for object recognition. DPT aims to rival traditional data-hungry pre-training techniques while avoiding the introduction of unnecessary features that can mislead the network in downstream tasks with distinct and scarce data. The preprint for this implementation can be found here: https://arxiv.org/abs/2312.00304

## Abstract

In response to the escalating data demands of DNNs for object recognition, we present Developmental PreTraining (DPT) as a potential solution. DPT is crafted to compete with traditional pre-training methods, focusing on avoiding unnecessary features that may mislead the network in downstream tasks with significantly different and limited data. The curriculum for DPT draws inspiration from human infant visual development, introducing carefully-selected primitive and universal features in a phased approach.

## Key Features

- Curriculum-based pre-training of a CNN
- Phased approach inspired by human infant visual development
- P1: Edges, P2: Shapes
- Evaluation against models with randomized weights
