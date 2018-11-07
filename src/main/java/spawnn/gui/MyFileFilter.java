package spawnn.gui;

import java.io.File;

import javax.swing.filechooser.FileFilter;

import org.apache.commons.io.FilenameUtils;

public abstract class MyFileFilter extends FileFilter {
	String ext;
	
	MyFileFilter(String ext) {
		this.ext = ext;
	}
	
	@Override
	public boolean accept(File f) {
		return f.isDirectory() || FilenameUtils.getExtension(f.getName()).equalsIgnoreCase(ext);
	}
	
	public File addExtension(File file) {
		if (!FilenameUtils.getExtension(file.getName()).equalsIgnoreCase(ext))
		    return new File(file.toString() + "." + ext ); 
		return file;
	}
}
