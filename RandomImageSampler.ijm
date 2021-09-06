setBatchMode(true);

parentDir = "Z:/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression/";
outputDir = "C:/Users/barryd/GitRepos/Python/Keras_Zebrafish_Classifier/test_data/";

for (i = 5; i <= 50; i++) {
	files = getFileList(parentDir + i);
	fileIndex = round((files.length - 1) * random());
	print("Opening " + parentDir + i + File.separator() + files[fileIndex]);
	open(parentDir + i + File.separator() + files[fileIndex]);
	print("Resizing...");
	run("Size...", "width=268 height=224 depth=1 constrain average interpolation=Bicubic");
	if(!File.exists(outputDir + i)){
		File.makeDirectory(outputDir + i);
	}
	print("Saving " + outputDir + i + File.separator() + files[fileIndex]);
	saveAs("PNG", outputDir + i + File.separator() + files[fileIndex]);
	close("*");
}

setBatchMode(false);

print("Done");