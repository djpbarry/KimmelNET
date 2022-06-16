setBatchMode(true);

parentDir = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression/";
outputDir = "C:/Users/barryd/GitRepos/Python/Zebrafish_Age_Estimator/test_data/";

indices = newArray(31, 23, 22, 46, 25);

for (i = 0; i < indices.length; i++) {
	files = getFileList(parentDir + indices[i]);
	fileIndex = round((files.length - 1) * random());
	print("Opening " + parentDir + indices[i] + File.separator() + files[fileIndex]);
	open(parentDir + indices[i] + File.separator() + files[fileIndex]);
	print("Resizing...");
	run("Size...", "width=268 height=224 depth=1 constrain average interpolation=Bicubic");
	if(!File.exists(outputDir + i)){
		File.makeDirectory(outputDir + indices[i]);
	}
	print("Saving " + outputDir + indices[i] + File.separator() + files[fileIndex]);
	saveAs("PNG", outputDir + indices[i] + File.separator() + files[fileIndex]);
	close("*");
}

setBatchMode(false);

print("Done");