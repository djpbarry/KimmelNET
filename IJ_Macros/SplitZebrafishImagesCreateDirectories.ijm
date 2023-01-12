setBatchMode(true);

initHPF = 4.5;
interval = 0.25;
maxSlices = 190;

paths = newArray("/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression_Downsized_with_BG_Noisy/",
				"/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression_Downsized_with_BG_Noisy/",
				"/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_25C_Downsized_with_BG_Noisy/");

outputFolders = newArray(maxSlices);

for(p = 0; p < paths.length; p++){
	if(File.exists(paths[p])){
		File.delete(paths[p]);
	}
	File.makeDirectory(paths[p]);
	for (o = 0; o < outputFolders.length; o++) {
		outputFolders[o] = initHPF + interval * o;
		File.makeDirectory(paths[p] + File.separator() + outputFolders[o]);
	}
}

setBatchMode(false);
