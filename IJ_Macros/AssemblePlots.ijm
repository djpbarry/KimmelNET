close("*");

inputDir = "Z:/working/barryd/hpc/python/zf_reg/outputs";

dirList = getFileList(inputDir);

index = 0;

for (i = 0; i < dirList.length; i++) {
	if(startsWith(dirList[i], "simple_regression_multi_gpu_custom_augmentation_2023-06")){
		print((index++) + ": " + dirList[i]);
		open(inputDir + File.separator() + dirList[i] + "plots" + File.separator() + "Princeton_Prediction_Accuracy.png");
		rename(dirList[i]);
	}
}

run("Images to Stack", "use");