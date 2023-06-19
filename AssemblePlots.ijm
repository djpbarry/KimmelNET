close("*");

inputDir = "Z:/working/barryd/hpc/python/zf_reg/outputs";

dirList = getFileList(inputDir);

for (i = 0; i < dirList.length; i++) {
	if(startsWith(dirList[i], "simple_regression")){
		open(inputDir + File.separator() + dirList[i] + "plots" + File.separator() + "Crick_Prediction_Accuracy.png");
		rename(dirList[i]);
	}
}

run("Images to Stack", "use");