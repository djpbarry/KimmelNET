input = "/camp/stp/lm/inputs/smithj/Rebecca Ann Jones/FishDev_WT_25C_1";
temp = "/camp/stp/lm/working/barryd/temp";
output = "/camp/stp/lm/working/barryd/hpc/python/zf_reg";

fileList = getFileList(input);

fileList = Array.deleteValue(fileList, "DisplaySettings.json");

File.makeDirectory(temp);

column = 0;
row = 0;

setBatchMode(true);

for (i = 0; i < fileList.length; i++) {
	print("Processing " + fileList[i]);
	open(input + File.separator() + fileList[i]);
	run("Size...", "width=77 height=64 depth=252 constrain average interpolation=Bilinear");
	run("Specify...", "width=64 height=64 x=38 y=32 slice=1 centered");
	run("Crop");
	current = getTitle();
	if(row > 0){
		run("Combine...", "stack1=[" + title + "] stack2=[" + current + "] combine");
		if(row == 11){
			saveAs("TIF", temp + File.separator() + "column_" + column++);
			close("*");
			row = 0;
		} else {
			title = getTitle();
			row++;
		}
	} else {
		title = getTitle();
		row++;
	}
}

tempFileList = getFileList(temp);
column = 0;

for (i = 0; i < tempFileList.length; i++) {
	open(temp + File.separator() + tempFileList [i]);
	current = getTitle();
	if(column > 0){
		run("Combine...", "stack1=[" + title + "] stack2=[" + current + "]");
	}
	title = getTitle();
	column++;
}

saveAs("TIF", output + File.separator() + File.getName(input));

setBatchMode(false);

//File.delete(temp);