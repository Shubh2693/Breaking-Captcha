#!/bin/bash

colored="./color_generated_captcha_images"

	for image in "$colored"/*
	do
		echo "$image"
		convert "$image" -threshold 80% "./color_generated_captcha_images/$(basename $image)"
	done
