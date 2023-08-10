setBatchMode(true);

run("Bio-Formats Macro Extensions");
setOption("ScaleConversions", true);

initHPF = 3.5;
interval = 0.25;
maxSlices = 180;

Dialog.create("Organise Images");
Dialog.addFile("Input file", getDirectory("current"));
Dialog.addDirectory("Output Directory", getDirectory("current"));
Dialog.addNumber("Inital HPF", initHPF);
Dialog.addNumber("Frame rate (h)", interval);
Dialog.addNumber("Maximum number of frames", maxSlices);
Dialog.show();

input = Dialog.getString();
output = Dialog.getString();
initHPF = Dialog.getNumber();
interval = Dialog.getNumber();
maxSlices = Dialog.getNumber();

outputFolders = newArray(maxSlices);

if(File.exists(output)){
	File.delete(output);
}
File.makeDirectory(output);
for (o = 0; o < outputFolders.length; o++) {
	outputFolders[o] = initHPF + interval * o;
	File.makeDirectory(output + File.separator() + outputFolders[o]);
}

timeStampOutput("Input Dataset: " + input);

Ext.setId(input);
Ext.getCurrentFile(file);
Ext.getSeriesCount(seriesCount)
Ext.getSizeX(sizeX);
Ext.getSizeY(sizeY);
Ext.getSizeC(sizeC);
Ext.getSizeZ(sizeZ);
Ext.getSizeT(sizeT);
Ext.getImageCount(slices);
timeStampOutput("Input Dimensions: " + sizeX + " " + sizeY + " " + sizeC + " " + sizeZ + " " + sizeT + " " + seriesCount);

for (s = 0; s < seriesCount; s++) {
	Ext.setSeries(s);
	timeStampOutput("Processing series " + s + " of " + seriesCount);
	for (i = 0; i < slices; i++) {
		Ext.openImage("plane " + i, i);
		run("8-bit");
		timeStampOutput("Saving slice " + i + " of " + slices);
		saveAs("PNG", output + File.separator() + outputFolders[i] + File.separator() + File.getName(input) + "-" + s + "-" + i + ".png" );
	    close("*");
	    timeStampOutput(IJ.freeMemory());
	}
}
	
timeStampOutput("Done.");
	
setBatchMode(false);

function timeStampOutput(message){
	getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
	print(year + "-" + month + "-" + dayOfMonth + "-" + hour + ":" + minute + ":" + second + " " + message);
}
