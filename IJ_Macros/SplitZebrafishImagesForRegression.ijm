setBatchMode(true);

initHPF = 4.5;
interval = 0.25;
maxSlices = 190;
testTrainSplit = 0.5;
nTrainSlices = round(testTrainSplit * maxSlices);

//filenames = newArray("20201127_FishDev_WT_28.5_1",
//					"FishDev_WT_02_3",
//					"FishDev_WT_01_1");

filenames = newArray("FishDev_WT_25C_1");

input = "/camp/stp/lm/inputs/smithj/Rebecca Ann Jones/";

//paths = newArray("/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Train_Regression/",
//				"/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_Test_Regression/");

paths = newArray("/camp/stp/lm/working/barryd/hpc/python/keras_image_class/Zebrafish_25C/");

outputFolders = newArray(maxSlices);

for(p = 0; p < paths.length; p++){
	for (o = 0; o < outputFolders.length; o++) {
		outputFolders[o] = initHPF + interval * o;
	}
}

letters=newArray('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H');
wells=newArray(96);
for(i=1; i<=12; i++){
	for(l=0; l<=7; l++){
		wells[l*12+i-1]=letters[l] + i;
	}
}

wellIndex = getArgument();
print("Well Index: " + wellIndex);
well = wells[wellIndex];
//testOrTrain = Math.pow(-1, wellIndex) > 0;

//for(j = 0; j < 3; j++){
for(j = 0; j < 1; j++){

	print("Dataset " + filenames[j] + " - well " + well);

	open(input + File.separator() + filenames[j] + File.separator() + filenames[j] + "_MMStack_" + well + "-Site_0.ome.tif");

	//randomSlices = createRandomSample(nTrainSlices, maxSlices);
	
	for (i = 1; i <= maxSlices ; i++) {
		setSlice(i);
		//if(testOrTrain){
			//saveAs("PNG", paths[0] + File.separator() + outputFolders[i-1] + File.separator() + filenames[j] + "-" + well + "-" + i + ".png" );
			//print("Slice " + i + " saved as training data");
		//} else {
			saveAs("PNG", paths[0] + File.separator() + outputFolders[i-1] + File.separator() + filenames[j] + "-" + well + "-" + i + ".png" );
			print("Slice " + i + " saved as test data");
		//}
	}
	close("*");
}

setBatchMode(false);


function createRandomSample(sampleSize, maxVal){
	samples=Array.getSequence(sampleSize);
	Array.fill(samples, -1);
	for(i=0; i < sampleSize; i++){
		index = round(maxVal * random() + 1);
		while(arrayContains(samples, index)){
			index = round(maxVal * random() + 1);
		}
		samples[i] = index;
	}
	return samples;
}

function arrayContains(array, val){
	for(i=0; i<array.length; i++){
		if(array[i] == val) return true;
	}
	return false;
}
