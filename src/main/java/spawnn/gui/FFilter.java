package spawnn.gui;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class FFilter {

	public static final FileFilter somXMLFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".xml");
		}

		@Override
		public String getDescription() {
			return "somXML (*.xml)";
		}
	};
	
	public static final FileFilter ngXMLFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".xml");
		}

		@Override
		public String getDescription() {
			return "ngXML (*.xml)";
		}
	};

	public static final FileFilter pngFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".png");
		}

		@Override
		public String getDescription() {
			return "Portable Network Graphics (*.png)";
		}
	};
	
	public static final FileFilter epsFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".eps");
		}

		@Override
		public String getDescription() {
			return "Encapsulated PostScript (*.eps)";
		}
	};

	public static final FileFilter unitFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".unit");
		}

		@Override
		public String getDescription() {
			return "SOMLib unit (*.unit)";
		}
	};

	public static final FileFilter weightFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".unit");
		}

		@Override
		public String getDescription() {
			return "SOMLib weight (*.wgt)";
		}
	};

	public static final FileFilter graphMLFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".graphml");
		}

		@Override
		public String getDescription() {
			return "GraphMl (*.graphml)";
		}
	};
	
	public static final FileFilter shpFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {
			return f.isDirectory() || f.getName().toLowerCase().endsWith(".shp");
		}

		@Override
		public String getDescription() {
			return "ESRI shapefile (*.shp)";
		}
	};
	
	public static final FileFilter csvFilter = new FileFilter() {
		@Override
		public boolean accept(File f) {	return f.isDirectory() || f.getName().toLowerCase().endsWith(".csv"); }

		@Override
		public String getDescription() { return "Comma-separated values (*.csv)"; }
	};
}
