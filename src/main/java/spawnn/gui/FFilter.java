package spawnn.gui;

import java.io.File;

import javax.swing.filechooser.FileFilter;

public class FFilter {

	public static final FileFilter somXMLFilter = new MyFileFilter("xml") {
		@Override
		public String getDescription() {
			return "somXML (*.xml)";
		}
	};
	
	public static final FileFilter ngXMLFilter = new MyFileFilter("xml") {
		@Override
		public String getDescription() {
			return "ngXML (*.xml)";
		}
	};

	public static final FileFilter pngFilter = new MyFileFilter("png") {
		@Override
		public String getDescription() {
			return "Portable Network Graphics (*.png)";
		}
	};
	
	public static final FileFilter epsFilter = new MyFileFilter("eps") {
		@Override
		public String getDescription() {
			return "Encapsulated PostScript (*.eps)";
		}
	};

	public static final FileFilter unitFilter = new MyFileFilter("unit") {
		@Override
		public String getDescription() {
			return "SOMLib unit (*.unit)";
		}
	};

	public static final FileFilter weightFilter = new MyFileFilter("weight") {
		@Override
		public String getDescription() {
			return "SOMLib weight (*.wgt)";
		}
	};

	public static final FileFilter graphMLFilter = new MyFileFilter("graphml") {
		@Override
		public String getDescription() {
			return "GraphMl (*.graphml)";
		}
	};
	
	public static final FileFilter shpFilter = new MyFileFilter("shp") {
		@Override
		public String getDescription() {
			return "ESRI shapefile (*.shp)";
		}
	};
	
	public static final FileFilter csvFilter = new MyFileFilter("csv") {
		@Override
		public String getDescription() { return "Comma-separated values (*.csv)"; }
	};
}
