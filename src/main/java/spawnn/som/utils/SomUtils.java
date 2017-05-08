package spawnn.som.utils;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Arc2D;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.geotools.feature.DefaultFeatureCollection;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.map.FeatureLayer;
import org.geotools.map.MapContent;
import org.geotools.renderer.GTRenderer;
import org.geotools.renderer.lite.StreamingRenderer;
import org.geotools.styling.SLD;
import org.geotools.styling.StyleBuilder;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.input.SAXBuilder;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LineString;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import imageware.Builder;
import imageware.ImageWare;
import spawnn.dist.Dist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.Grid2DHexToroid;
import spawnn.som.grid.Grid2DToroid;
import spawnn.som.grid.GridPos;
import spawnn.utils.ColorBrewer;
import spawnn.utils.ColorUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import watershedflooding.Watershed;
import watershedflooding.WatershedMeasurements;

// TODO: lot/some of methods here can be maded generic (e.g. for use with SOM and NG), especially bmu-related stuff
// TODO replace all this double[][]-stuff by grid<>
public class SomUtils {

	// new quality measure
	public static double getMappedClassVariance(List<double[]> samples, Map<double[], Integer> classes, Grid<double[]> grid, BmuGetter<double[]> bg) {
		double totalSum = 0;

		Map<double[], GridPos> bmuMap = new HashMap<double[], GridPos>();
		for (double[] d : samples)
			bmuMap.put(d, bg.getBmuPos(d, grid));

		for (int cls : classes.values()) {
			List<GridPos> cluster = new ArrayList<GridPos>();

			// get neurons for this class
			for (double[] d : samples) {
				GridPos bmu = bmuMap.get(d);

				if (classes.get(d) == cls)
					cluster.add(bmu);
			}

			// get center
			double[] c = new double[cluster.get(0).length()];

			for (GridPos p : cluster) {
				for (int i = 0; i < c.length; i++)
					c[i] += p.getPos(i);
			}
			int[] center = new int[c.length];
			for (int i = 0; i < c.length; i++)
				center[i] = (int) Math.round(c[i] / cluster.size());

			double distSum = 0;
			for (GridPos p : cluster) {
				distSum += Math.pow(grid.dist(p, new GridPos(center)), 2);
			}

			totalSum += distSum / cluster.size();
		}
		return totalSum;
	}

	public static <T> double getEntropy(List<double[]> samples, Map<T, Set<double[]>> bmus) {
		double entropy = 0;
		for (T gp : bmus.keySet()) {
			if (bmus.containsKey(gp) && !bmus.get(gp).isEmpty()) {
				double d = (double) bmus.get(gp).size() / samples.size();
				entropy += -d * Math.log(d) / Math.log(2);
			}
		}
		return entropy;
	}

	public static Collection<Set<GridPos>> getWatershedHex(int mi, int ma, double smooth, Grid2D<double[]> grid, Dist<double[]> d, boolean debug) {
		double[][] umatrix = getNormedMatrix(getUMatrix(grid, d));

		int xDiff = 12;
		int yDiff = 14;

		int xSize = (8 * umatrix.length + 4 * (umatrix.length + 1));
		int ySize = 14 * umatrix[0].length + 14 + 1;

		BufferedImage bi = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bi.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		for (int i = 0; i < umatrix.length; i++) {
			for (int j = 0; j < umatrix[i].length; j++) {

				float v = (float) umatrix[i][j];

				int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
				int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

				if (i % 2 == 1)
					yc += yDiff * 1.0 / 2;

				if ((i + 2) % 4 == 0)
					yc += yDiff;

				int[] x = { xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
				int[] y = { yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };

				g2.setColor(new Color(1 - v, 1 - v, 1 - v));
				g2.fill(new Polygon(x, y, x.length));
			}
		}
		g2.dispose();

		// create a grayscale image the same size
		BufferedImage gray = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

		// convert the original colored image to grayscale
		ColorConvertOp op = new ColorConvertOp(bi.getColorModel().getColorSpace(), gray.getColorModel().getColorSpace(), null);
		op.filter(bi, gray);

		ImageProcessor ip = new ByteProcessor(gray);

		String title = "umatrix";
		ImagePlus imp = new ImagePlus(title, ip);

		// IJ.save(imp, "output/umatrix.bmp");
		// IJ.runPlugIn(imp, "Watershed ", "");

		// imp.show("umatrix");

		ImageWare image = Builder.wrap(imp);
		image.invert();
		image.smoothGaussian(smooth);

		Watershed watershed = new Watershed(false); // show progressen status in console?
		watershed.doWatershed(image, true, mi, ma); // false a neighborhood of 4 pixels, true a neighborhood of 8 pixels
		// floods from mi to ma?

		if (debug) {
			ImagePlus dams = watershed.getDams();
			dams.show("dams");
			ImagePlus reddams = watershed.getRedDams(imp);
			reddams.show("redams");
			ImagePlus composite = watershed.getComposite(imp);
			composite.show("composite");
			WatershedMeasurements.measure(composite);
		}

		ImagePlus basins = watershed.getBasins();
		int[][] img = basins.getProcessor().getIntArray();

		Map<Integer, Set<GridPos>> cluster = new HashMap<Integer, Set<GridPos>>();
		for (int i = 0; i < umatrix.length; i++) {
			for (int j = 0; j < umatrix[i].length; j++) {

				// continue if intermediate cell
				if (i % 2 != 0 || j % 2 != 0)
					continue;

				int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
				int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

				if (i % 2 == 1)
					yc += yDiff * 1.0 / 2;

				if ((i + 2) % 4 == 0)
					yc += yDiff;

				int col = img[xc][yc]; // color in image

				if (!cluster.containsKey(col))
					cluster.put(col, new HashSet<GridPos>());
				cluster.get(col).add(new GridPos(i / 2, j / 2));
			}
		}
		return cluster.values();
	}

	// dist required for u-matrix, TODO: make it useable for any matrix (d or u), Problem: hexmatrix is not adequatly treated by the method
	public static int[][] getWatershed(int mi, int ma, double smooth, Grid2D<double[]> grid, Dist<double[]> d, boolean debug) {
		double[][] umatrix = getUMatrix(grid, d);
		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0; i < umatrix.length; i++)
			for (int j = 0; j < umatrix[0].length; j++) {
				if (umatrix[i][j] > max)
					max = umatrix[i][j];
				if (umatrix[i][j] < min)
					min = umatrix[i][j];
			}

		// TODO: min verwenden?
		// min = 0;

		byte[] pixels = new byte[(umatrix.length + 2) * (umatrix[0].length + 2)];
		for (int i = 1; i < umatrix.length + 1; i++)
			for (int j = 1; j < umatrix[0].length + 1; j++)
				pixels[i + j * (umatrix.length + 2)] = (byte) (255 * (umatrix[i - 1][j - 1] - min) / (max - min));

		ImageProcessor ip = new ByteProcessor(umatrix.length + 2, umatrix[0].length + 2);
		ip.setPixels(pixels);

		String title = "umatrix";
		ImagePlus imp = new ImagePlus(title, ip);

		// IJ.save(imp, "output/umatrix.bmp");
		// IJ.runPlugIn(imp, "Watershed ", "");

		// imp.show("umatrix");

		ImageWare image = Builder.wrap(imp);
		image.smoothGaussian(smooth);

		Watershed watershed = new Watershed(false); // show progressen status in console?
		watershed.doWatershed(image, true, mi, ma); // false a neighborhood of 4 pixels, true a neighborhood of 8 pixels

		if (debug) {
			ImagePlus dams = watershed.getDams();
			dams.show("dams");
			ImagePlus reddams = watershed.getRedDams(imp);
			reddams.show("redams");
			ImagePlus composite = watershed.getComposite(imp);
			composite.show("composite");
			WatershedMeasurements.measure(composite);
		}

		ImagePlus basins = watershed.getBasins();
		int[][] img = basins.getProcessor().getIntArray();

		// restore orig size
		int border = img[0][0]; // border color
		int[][] nImg = new int[img.length - 2][img[0].length - 2];
		for (int i = 1; i < img.length - 1; i++)
			for (int j = 1; j < img[0].length - 1; j++)
				nImg[i - 1][j - 1] = img[i][j];
		img = nImg;

		// replace "border" by nearest color
		nImg = new int[img.length][img[0].length];
		for (int i = 0; i < img.length; i++) {
			for (int j = 0; j < img[0].length; j++) {

				if (img[i][j] != border) {
					nImg[i][j] = img[i][j];
					continue;
				}

				// search in umatrix for nearest
				double minDist = Double.MAX_VALUE;
				int cl = border;

				List<int[]> unbs = new ArrayList<int[]>();
				if (i > 0 && j > 0)
					unbs.add(new int[] { i - 1, j - 1 });
				if (i > 0)
					unbs.add(new int[] { i - 1, j });
				if (i > 0 && j < img[0].length - 1)
					unbs.add(new int[] { i - 1, j + 1 });
				if (j > 0)
					unbs.add(new int[] { i, j - 1 });
				if (i < img.length - 1 && j < img[0].length - 1)
					unbs.add(new int[] { i + 1, j + 1 });
				if (i < img.length - 1)
					unbs.add(new int[] { i + 1, j });
				if (i < img.length - 1 && j > 0)
					unbs.add(new int[] { i + 1, j - 1 });
				if (j < img[0].length - 1)
					unbs.add(new int[] { i, j + 1 - 1 });

				for (int[] p : unbs) {
					if (cl == border || (img[p[0]][p[1]] != border && umatrix[p[0]][p[1]] < umatrix[i][j] && Math.abs(umatrix[i][j] - umatrix[p[0]][p[1]]) < minDist)) {
						minDist = Math.abs(umatrix[i][j] - umatrix[p[0]][p[1]]);
						cl = img[p[0]][p[1]];
					}
				}
				nImg[i][j] = cl;
			}
		}

		return img;
	}

	public static Collection<Set<GridPos>> getClusterFromWatershed(int[][] nImg, Grid<double[]> grid) {
		// build clusters from image
		Map<Integer, Set<GridPos>> cluster = new HashMap<Integer, Set<GridPos>>();
		for (int i = 0; i < nImg.length; i++) {
			for (int j = 0; j < nImg[0].length; j++) {

				if (i % 2 != 0 || j % 2 != 0)
					continue;

				GridPos pos = null; // actual pos
				for (GridPos p : grid.getPositions())
					if (p.getPos(0) == i / 2 && p.getPos(1) == j / 2)
						pos = p;

				int cl = nImg[i][j];
				if (cluster.containsKey(cl)) {
					cluster.get(cl).add(pos);
				} else {
					Set<GridPos> l = new HashSet<GridPos>();
					l.add(pos);
					cluster.put(cl, l);
				}
			}
		}
		return cluster.values();
	}
	
	@Deprecated
	public static Map<GridPos, Set<double[]>> getBmuMapping(List<double[]> samples, Grid<double[]> grid, BmuGetter<double[]> b ) {
		return getBmuMapping( samples, grid, b, true );
	}

	public static Map<GridPos, Set<double[]>> getBmuMapping(List<double[]> samples, Grid<double[]> grid, BmuGetter<double[]> b, boolean includeEmpty ) {
		Map<GridPos, Set<double[]>> r = new HashMap<GridPos, Set<double[]>>();	
		for (double[] s : samples) {
			GridPos bmu = b.getBmuPos(s, grid);
			if( !r.containsKey(bmu) )
				r.put(bmu, new HashSet<double[]>() );
			r.get(bmu).add(s);
		}
		if( includeEmpty )
			for( GridPos p : grid.getPositions() )
				if( !r.containsKey(p) )
					r.put(p, new HashSet<double[]>() );
		return r;
	}

	public static void printClassDist(Collection<Set<double[]>> cluster, Map<GridPos, Set<double[]>> mapping, Grid2D<double[]> grid, String fn) {

		// assign to each sample a cluster-number
		Map<double[], Integer> classMap = new HashMap<double[], Integer>();
		int i = 0;
		for (Set<double[]> s : cluster) {
			for (double[] d : s)
				classMap.put(d, i);
			i++;
		}
		try {
			printClassDist(classMap, mapping, grid, new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void printClassDist(Map<double[], Integer> classes, Map<GridPos, Set<double[]>> mapping, Grid2D<double[]> grid, String fn) {
		try {
			printClassDist(classes, mapping, grid, new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	@Deprecated
	public static void printClassDist(Map<double[], Integer> classes, Map<GridPos, Set<double[]>> mapping, Grid2D<double[]> grid, OutputStream os) {
		if (grid instanceof Grid2DHex)
			printImage(getHexClassDistImage(getClassDist(classes, mapping, grid), (Grid2DHex<double[]>) grid, 5), os);
		else
			printImage(getClassDistImage(getClassDist(classes, mapping, grid), grid, 50), os);
	}

	public static Map<GridPos, Map<Integer, Integer>> getClassDist(Map<double[], Integer> classes, Map<GridPos, Set<double[]>> mapping, Grid<double[]> grid) {
		Map<GridPos, Map<Integer, Integer>> pcc = new HashMap<GridPos, Map<Integer, Integer>>();
		for (GridPos p : grid.getPositions()) {
			// get classes and occurence for cur position
			Map<Integer, Integer> classCount = new HashMap<Integer, Integer>();
			if (mapping.containsKey(p)) {
				for (double[] d : mapping.get(p)) {
					int c = classes.get(d);
					if (classCount.containsKey(c))
						classCount.put(c, classCount.get(c) + 1);
					else
						classCount.put(c, 1);
				}
			}
			pcc.put(p, classCount);
		}
		return pcc;
	}

	public static BufferedImage getClassDistImage(Map<GridPos, Map<Integer, Integer>> pcc, Grid2D<double[]> grid, int cellSize) {
		int xDim = grid.getSizeOfDim(0);
		int yDim = grid.getSizeOfDim(1);

		BufferedImage bufImg = new BufferedImage(cellSize * xDim, cellSize * yDim, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		for (GridPos p : pcc.keySet()) {
			int i = p.getPos(0);
			int j = p.getPos(1);

			g2.setColor(Color.BLACK);
			g2.drawRect(i * cellSize, j * cellSize, cellSize, cellSize);

			Map<Integer, Integer> classMap = pcc.get(p);

			if (classMap.isEmpty())
				continue;

			List<Integer> sortedClasses = new ArrayList<Integer>(classMap.keySet());
			Collections.sort(sortedClasses);

			int sum = 0;
			for (int v : classMap.values())
				sum += v;

			int startAngle = 0;
			for (int c : sortedClasses) {
				int angle = Math.round((360 * classMap.get(c)) / sum);
				g2.setColor(Drawer.getColor(c));
				g2.fillArc(i * cellSize, j * cellSize, cellSize, cellSize, startAngle, angle);
				startAngle += angle;
			}
		}
		g2.dispose();
		return bufImg;
	}

	public static BufferedImage getHexClassDistImage(Map<GridPos, Map<Integer, Integer>> pcc, Grid2DHex<double[]> grid, int scale) {
		int xDiff = 12;
		int yDiff = 14;

		int xDim = grid.getSizeOfDim(0);
		int yDim = grid.getSizeOfDim(1);

		BufferedImage bufImg = new BufferedImage((8 * xDim + 4 * (xDim + 1)) * scale, 14 * yDim * scale + 7 * scale + 1, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		for (GridPos pos : pcc.keySet()) {
			int i = pos.getPos(0);
			int j = pos.getPos(1);
			int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
			int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

			if (i % 2 == 1)
				yc += yDiff * 1.0 / 2;

			// (4|7), P3 (8|0), P5 (4|−7), P7 (−4|−7), P9 (−8|0), P11 (−4|7)
			int[] x = { xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
			int[] y = { yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };

			Polygon p = new Polygon(x, y, x.length);

			Map<Integer, Integer> classMap = pcc.get(pos);
			if (!classMap.isEmpty()) {

				List<Integer> sortedClasses = new ArrayList<Integer>(classMap.keySet());
				Collections.sort(sortedClasses);

				int sum = 0;
				for (int v : classMap.values())
					sum += v;

				// g2.setClip(sp);

				int startAngle = 0;
				for (int c : sortedClasses) {
					int angle = Math.round((360 * classMap.get(c)) / sum);
					Shape arc = new Arc2D.Double(xc - 6, yc - 6, 12, 12, startAngle, angle, Arc2D.PIE); // kreise zu klein
					// Shape arc = new Arc2D.Double( xc-7, yc-7, 14, 14, startAngle, angle, Arc2D.PIE ); // kreise zu groß
					// Shape arc = new Arc2D.Double( xc-6, yc-7, 12, 14, startAngle, angle, Arc2D.PIE ); // kreise eliptisch
					AffineTransform at2 = new AffineTransform();
					at2.scale(scale, scale);

					// circle
					Shape sp2 = at2.createTransformedShape(arc);
					g2.setColor(Drawer.getColor(c));
					g2.fill(sp2);
					// g2.draw(sp);

					startAngle += angle;
				}

				g2.setColor(Color.BLACK);
				g2.drawString(sum + "", (xc - 4) * scale, yc * scale);
			}

			AffineTransform at = new AffineTransform();
			at.scale(scale, scale);
			Shape sp = at.createTransformedShape(p);
			g2.setColor(Color.BLACK);
			g2.draw(sp);
		}
		g2.dispose();
		return bufImg;
	}

	public static void printDMatrix(Grid2D<double[]> grid, Dist<double[]> d, String fn) {
		try {
			printDMatrix(grid, d, new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printDMatrix(Grid2D<double[]> grid, Dist<double[]> d, ColorBrewer cm, OutputStream os) {
		double[][] dmatrix = getDMatrix(grid, d);
		if (grid instanceof Grid2DHex)
			printImage(getHexMatrixImage(dmatrix, 5, cm, HEX_NORMAL), os);
		else
			printImage(getRectMatrixImage(dmatrix, 50, cm), os);
	}

	public static void printDMatrix(Grid2D<double[]> grid, Dist<double[]> d, OutputStream os) {
		printDMatrix(grid, d, ColorBrewer.Greys, os);
	}

	// Only used by watershed clustering... do we need it anyways?
	private static double[][] getNormedMatrix(double[][] m) {
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[i].length; j++) {
				min = Math.min(min, m[i][j]);
				max = Math.max(max, m[i][j]);
			}
		}

		double[][] mNorm = new double[m.length][m[0].length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[i].length; j++)
				mNorm[i][j] = (m[i][j] - min) / (max - min);

		return mNorm;
	}

	public static void printComponentPlane(Grid2D<double[]> grid, int idx, String fn) {
		try {
			printComponentPlane(grid, idx, new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printComponentPlane(Grid2D<double[]> grid, int idx, ColorBrewer colorMode, OutputStream os) {
		if (grid instanceof Grid2DHex)
			printImage(getHexMatrixImage((Grid2DHex<double[]>) grid, 5, colorMode, HEX_NORMAL, idx), os);
		else {
			double[][] cm = getComponentMatrix(grid, idx);
			printImage(getRectMatrixImage(cm, 50, colorMode), os);
		}
	}

	public static void printComponentPlane(Grid2D<double[]> grid, int idx, OutputStream os) {
		printComponentPlane(grid, idx, ColorBrewer.Greys, os);
	}

	public static void printUMatrix(Grid2D<double[]> grid, Dist<double[]> d, String file) {
		try {
			printUMatrix(grid, d, new FileOutputStream(file));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printUMatrix(Grid2D<double[]> grid, Dist<double[]> d, OutputStream os) {
		double[][] umatrix = getUMatrix(grid, d);

		if (grid instanceof Grid2DHex)
			printImage(getHexMatrixImage(umatrix, 5, ColorBrewer.Greys, HEX_UMAT), os);
		else
			printImage(getRectMatrixImage(umatrix, (int) (100 * grid.size() / (grid.size() * 2)), ColorBrewer.Greys), os);
	}

	public static void printHexUMat(Grid2D<double[]> grid, Dist<double[]> d, ColorBrewer colorScale, OutputStream os) {
		double[][] umatrix = getUMatrix(grid, d);
		printImage(getHexMatrixImage(umatrix, 5, colorScale, HEX_UMAT), os);
	}

	public static void printUMatrix(Grid2D<double[]> grid, Dist<double[]> d, ColorBrewer cm, boolean rectType, String fn) {
		double[][] umatrix = getUMatrix(grid, d);

		try {
			FileOutputStream os = new FileOutputStream(fn);

			if (grid instanceof Grid2DHex)
				printImage(getHexMatrixImage(umatrix, 5, cm, rectType), os);
			else
				// type ignored for other grids
				printImage(getRectMatrixImage(umatrix, (int) (100 * grid.size() / (grid.size() * 2)), cm), os);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void printImage(BufferedImage img, OutputStream os) {
		try {
			ImageIO.write(img, "png", os);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static BufferedImage getRectMatrixImage(double[][] matrix, int cellSize, ColorBrewer colorScale) {
		BufferedImage bufImg = new BufferedImage(cellSize * matrix.length, cellSize * matrix[0].length, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Map<GridPos, Double> mMap = new HashMap<GridPos, Double>();
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++)
				mMap.put(new GridPos(i, j), matrix[i][j]);
		Map<GridPos, Color> cMap = ColorUtils.getColorMap(mMap, colorScale);

		for (GridPos p : cMap.keySet()) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			g2.setColor(cMap.get(p));
			g2.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
		}

		g2.dispose();
		return bufImg;
	}

	public static boolean HEX_UMAT = true, HEX_NORMAL = false;

	@Deprecated
	public static BufferedImage getHexMatrixImage(double[][] matrix, int scale, ColorBrewer colorScale, boolean umat) {
		int xDiff = 12;
		int yDiff = 14;

		int xSize = (8 * matrix.length + 4 * (matrix.length + 1)) * scale;
		int ySize;
		if (umat == HEX_UMAT)
			ySize = 14 * matrix[0].length * scale + 14 * scale + 1;
		else
			ySize = 14 * matrix[0].length * scale + 7 * scale + 1;

		BufferedImage bufImg = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Map<Shape, Double> shapes = new HashMap<Shape, Double>();

		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
				int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

				if (i % 2 == 1)
					yc += yDiff * 1.0 / 2;

				if ((i + 2) % 4 == 0 && umat == HEX_UMAT)
					yc += yDiff;

				// (4|7), P3 (8|0), P5 (4|−7), P7 (−4|−7), P9 (−8|0), P11 (−4|7)
				int[] x = { xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
				int[] y = { yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };

				Polygon p = new Polygon(x, y, x.length);

				AffineTransform at = new AffineTransform();
				at.scale(scale, scale);
				Shape sp = at.createTransformedShape(p);
				shapes.put(sp, matrix[i][j]);
			}
		}
		Map<Shape, Color> colors = ColorUtils.getColorMap(shapes, colorScale);

		// fill hexagons
		DecimalFormat df = new DecimalFormat("#.#####");
		for (Shape sp : colors.keySet()) {
			g2.setColor(colors.get(sp));
			g2.fill(sp);

			Rectangle r = sp.getBounds();
			g2.setColor(Color.BLACK);
			g2.drawString(df.format(shapes.get(sp)) + "", (int) r.getMinX() + 5, (int) r.getCenterY());
		}

		// draw outlines
		for (Shape sp : colors.keySet()) {
			g2.setColor(Color.BLACK);
			g2.draw(sp);
		}

		g2.dispose();
		return bufImg;
	}

	// FIXME not good, better: getShapes(rect/hex) and then combine with colors
	public static BufferedImage getHexMatrixImage(Grid2DHex<double[]> grid, int scale, ColorBrewer cm, boolean type, int idx) {
		int xDiff = 12;
		int yDiff = 14;

		int xDim = grid.getSizeOfDim(0);
		int yDim = grid.getSizeOfDim(1);

		int xSize = (8 * xDim + 4 * (xDim + 1)) * scale;
		int ySize;
		if (type == HEX_UMAT)
			ySize = 14 * yDim * scale + 14 * scale + 1;
		else
			ySize = 14 * yDim * scale + 7 * scale + 1;

		BufferedImage bufImg = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Map<Shape, Double> shapes = new HashMap<Shape, Double>();

		for (GridPos gp : grid.getPositions()) {
			int i = gp.getPos(0);
			int j = gp.getPos(1);
			int xc = i * xDiff + (int) (2 * xDiff * 1.0 / 3);
			int yc = j * yDiff + (int) (yDiff * 1.0 / 2);

			if (i % 2 == 1)
				yc += yDiff * 1.0 / 2;

			if ((i + 2) % 4 == 0 && type == HEX_UMAT)
				yc += yDiff;

			// (4|7), P3 (8|0), P5 (4|−7), P7 (−4|−7), P9 (−8|0), P11 (−4|7)
			int[] x = { xc + 4, xc + 8, xc + 4, xc - 4, xc - 8, xc - 4, xc + 4 };
			int[] y = { yc + 7, yc + 0, yc - 7, yc - 7, yc + 0, yc + 7, yc + 7 };

			Polygon p = new Polygon(x, y, x.length);

			AffineTransform at = new AffineTransform();
			at.scale(scale, scale);
			Shape sp = at.createTransformedShape(p);
			shapes.put(sp, grid.getPrototypeAt(gp)[idx]);
		}
		Map<Shape, Color> colors = ColorUtils.getColorMap(shapes, cm, ColorUtils.ColorClass.Equal );

		// fill hexagons
		DecimalFormat df = new DecimalFormat("#.#####");
		for (Shape sp : colors.keySet()) {
			g2.setColor(colors.get(sp));
			g2.fill(sp);

			Rectangle r = sp.getBounds();
			g2.setColor(Color.BLACK);
			g2.drawString(df.format(shapes.get(sp)) + "", (int) r.getMinX() + 5, (int) r.getCenterY());
		}

		// draw outlines
		for (Shape sp : colors.keySet()) {
			g2.setColor(Color.BLACK);
			g2.draw(sp);
		}

		g2.dispose();
		return bufImg;
	}

	public static BufferedImage getImageWithPaths(Grid2DHex<double[]> grid, Collection<GridPos[]> c, int scale) {
		BufferedImage bufImg = new BufferedImage((8 * grid.getSizeOfDim(0) + 4 * (grid.getSizeOfDim(0) + 1)) * scale, 14 * grid.getSizeOfDim(1) * scale + 7 * scale + 1, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		int i = 0;
		for (GridPos[] ps : c) {
			Color col = Drawer.getColor(i++);

			for (int j = 1; j < ps.length; j++) {
				Line2D.Double l = new Line2D.Double(ps[j - 1].getPos(0), ps[j - 1].getPos(1), ps[j].getPos(0), ps[j].getPos(1));

				AffineTransform at = new AffineTransform();
				at.scale(scale, scale);
				Shape sp = at.createTransformedShape(l);
				g2.setColor(col);
				g2.draw(sp);
			}
		}

		g2.dispose();
		return bufImg;
	}

	public static double[][] getUMatrix(Grid2D<double[]> grid, Dist<double[]> d) {
		double[][] umatrix = new double[grid.getSizeOfDim(0) * 2 - 1][grid.getSizeOfDim(1) * 2 - 1];

		for (GridPos p : grid.getPositions()) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			double[] v = grid.getPrototypeAt(p);

			// average sum of differences to all neighbors
			double height = 0;
			for (GridPos nb : grid.getNeighbours(p)) {
				double[] nv = grid.getPrototypeAt(nb);
				height += d.dist(v, nv);
			}
			umatrix[2 * i][2 * j] = height / grid.getNeighbours(p).size();

			for (GridPos nb : grid.getNeighbours(p)) {

				int ni = nb.getPos(0);
				int nj = nb.getPos(1);
				double[] nv = grid.getPrototypeAt(nb);

				umatrix[2 * i + (ni - i)][2 * j + (nj - j)] = d.dist(nv, v);
			}

			if (!(grid instanceof Grid2DHex)) {

				if (i - 1 >= 0 && j - 1 >= 0) {
					double[] ol = grid.getPrototypeAt(new GridPos(i - 1, j - 1));
					umatrix[2 * i - 1][2 * j - 1] += 0.25 * d.dist(ol, v);
				}

				if (i + 1 < grid.getSizeOfDim(0) && j - 1 >= 0) {
					double[] or = grid.getPrototypeAt(new GridPos(i + 1, j - 1));
					umatrix[2 * i + 1][2 * j - 1] += 0.25 * d.dist(or, v);
				}

				if (i - 1 >= 0 && j + 1 < grid.getSizeOfDim(1)) {
					double[] ul = grid.getPrototypeAt(new GridPos(i - 1, j + 1));
					umatrix[2 * i - 1][2 * j + 1] += 0.25 * d.dist(ul, v);
				}

				if (i + 1 < grid.getSizeOfDim(0) && j + 1 < grid.getSizeOfDim(1)) {
					double[] ur = grid.getPrototypeAt(new GridPos(i + 1, j + 1));
					umatrix[2 * i + 1][2 * j + 1] += 0.25 * d.dist(ur, v);
				}
			}
		}
		return umatrix;
	}

	public static double[][] getDMatrix(Grid2D<double[]> grid, Dist<double[]> d) {

		double[][] dmatrix = new double[grid.getSizeOfDim(0)][grid.getSizeOfDim(1)];

		for (GridPos p : grid.getPositions()) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			double[] v = grid.getPrototypeAt(p);

			double height = 0;
			for (GridPos np : grid.getNeighbours(p))
				height += d.dist(v, grid.getPrototypeAt(np));
			dmatrix[i][j] = height / grid.getNeighbours(p).size();
		}
		return dmatrix;
	}

	public static double[][] getComponentMatrix(Grid2D<double[]> grid, int idx) {
		double[][] componentMatrix = new double[grid.getSizeOfDim(0)][grid.getSizeOfDim(1)];
		for (GridPos p : grid.getPositions()) {
			int i = p.getPos(0);
			int j = p.getPos(1);
			double[] v = grid.getPrototypeAt(p);
			componentMatrix[i][j] = v[idx];
		}
		return componentMatrix;
	}

	public static void printPositions(Grid2D<double[]> grid, int[] geocoords, String fn, int CELLSIZE) {
		try {
			printPositions(grid, geocoords, new FileOutputStream(fn), CELLSIZE);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void printPositions(Grid2D<double[]> grid, int[] geocoords, OutputStream os, int CELLSIZE) {

		BufferedImage bufImg = new BufferedImage(grid.getSizeOfDim(0) * CELLSIZE, grid.getSizeOfDim(1) * CELLSIZE, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (GridPos p : grid.getPositions()) {
			double[] v = grid.getPrototypeAt(p);
			double x = v[geocoords[0]];
			double y = v[geocoords[1]];

			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;
		}

		g2.setColor(Color.WHITE);
		for (double[] v : grid.getPrototypes()) {
			int x = (int) (bufImg.getWidth() * (v[0] - minX) / (maxX - minX));
			int y = (int) (bufImg.getHeight() * (v[1] - minY) / (maxY - minY));
			g2.fillOval(x, y, 5, 5);
		}

		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", os);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static void printBacaoMap(Map<GridPos, Set<double[]>> mapping, Dist<double[]> d, Grid2D<double[]> grid, int[] geocoords, OutputStream os) {
		int xScale = 1000;
		int yScale = 800;
		GeometryFactory gf = new GeometryFactory();

		BufferedImage bufImg = new BufferedImage(xScale, yScale, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		Map<Coordinate, GridPos> coords = new HashMap<Coordinate, GridPos>();
		Map<GridPos, Double> qe = new HashMap<GridPos, Double>();
		double maxQe = Double.MIN_VALUE;

		// for scaling
		double maxX = Double.MIN_VALUE;
		double minX = Double.MAX_VALUE;
		double maxY = Double.MIN_VALUE;
		double minY = Double.MAX_VALUE;

		for (GridPos p : grid.getPositions()) {
			double[] v = grid.getPrototypeAt(p);
			double x = v[geocoords[0]];
			double y = v[geocoords[1]];

			if (x > maxX)
				maxX = x;
			if (y > maxY)
				maxY = y;
			if (x < minX)
				minX = x;
			if (y < minY)
				minY = y;

			coords.put(new Coordinate(x, y), p);

			double err = 0;
			for (double[] s : mapping.get(p))
				err += d.dist(s, v);
			err /= mapping.get(p).size();

			if (err > maxQe)
				maxQe = err;

			qe.put(p, err);
		}

		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords.keySet());

		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		for (int i = 0; i < coll.getNumGeometries(); i++) {

			Geometry g = coll.getGeometryN(i);

			double err = -1;
			for (Coordinate c : coords.keySet()) {
				if (!g.contains(gf.createPoint(c)))
					continue;
				err = qe.get(coords.get(c));
			}

			if (err >= 0)
				g2.setColor(new Color((float) (1 - err / maxQe), (float) (1 - err / maxQe), (float) (1 - err / maxQe)));
			else
				g2.setColor(Color.RED);

			Polygon p = new Polygon();
			for (Coordinate c : g.getCoordinates()) {
				int x = (int) (xScale * (c.x - minX) / (maxX - minX));
				int y = (int) (yScale * (c.y - minY) / (maxY - minY));
				p.addPoint(x, y);
			}
			g2.fillPolygon(p);
		}

		// print network
		g2.setColor(new Color(0, 100, 0));
		for (GridPos a : mapping.keySet()) {
			double[] av = grid.getPrototypeAt(a);
			int x1 = (int) (xScale * (av[geocoords[0]] - minX) / (maxX - minX));
			int y1 = (int) (yScale * (av[geocoords[1]] - minY) / (maxY - minY));

			for (GridPos b : grid.getNeighbours(a)) {
				double[] bv = grid.getPrototypeAt(b);
				int x2 = (int) (xScale * (bv[geocoords[0]] - minX) / (maxX - minX));
				int y2 = (int) (yScale * (bv[geocoords[1]] - minY) / (maxY - minY));

				g2.drawLine(x1, y1, x2, y2);
			}
		}

		// print neurons pos on top
		for (GridPos p : mapping.keySet()) {
			double[] v = grid.getPrototypeAt(p);
			g2.setColor(Color.GREEN);
			int x = (int) (xScale * (v[geocoords[0]] - minX) / (maxX - minX));
			int y = (int) (yScale * (v[geocoords[1]] - minY) / (maxY - minY));
			g2.drawOval(x, y, 2, 2);
			double e = Math.round(qe.get(p) * 1000) / 1000.0;
			;
			g2.drawString(e + "", x, y);

			for (int i = 2; i < grid.getPrototypeAt(p).length; i++) {
				double a = Math.round(grid.getPrototypeAt(p)[i] * 1000) / 1000.0;
				;
				g2.drawString(a + "", x, y + ((i - 1) * 10));
			}
		}

		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", os);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	// TODO Rework plz
	public static void printClusters(Collection<Set<GridPos>> clusters, Grid2D<double[]> grid, OutputStream os) {
		int CELL_SIZE = 50;
		int xDim = grid.getSizeOfDim(0);
		int yDim = grid.getSizeOfDim(1);

		BufferedImage bufImg = new BufferedImage(CELL_SIZE * xDim, CELL_SIZE * yDim, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		if (clusters.size() > 19)
			System.err.println("Too many clusters!");

		int i = 0;
		for (Set<GridPos> cluster : clusters) {
			g2.setColor(Drawer.getColor(i++ % 19));
			for (GridPos p : cluster)
				g2.fillRect(p.getPos(0) * CELL_SIZE, p.getPos(1) * CELL_SIZE, CELL_SIZE, CELL_SIZE);
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", os);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	// see e.g. http://www.ica.luz.ve/~enava/redesn/ebooks/TopologyPreservationInSOM.pdf
	public static <T> double getMeanQuantError(Grid<double[]> grid, BmuGetter<double[]> bmuGetter, Dist<double[]> d, List<double[]> samples) {
		Map<GridPos, Set<double[]>> mapping = getBmuMapping(samples, grid, bmuGetter);
		Map<double[], Set<double[]>> clusters = new HashMap<double[], Set<double[]>>();
		for (GridPos k : mapping.keySet())
			clusters.put(grid.getPrototypeAt(k), mapping.get(k));
		return DataUtils.getMeanQuantizationError(clusters, d);
	}
	
	public static <T> double getMeanQuantError(Grid<double[]> grid, Map<GridPos,Set<double[]>> bmus, Dist<double[]> dist ) {
		Map<double[], Set<double[]>> dBmus = new HashMap<>();
		for (Entry<GridPos, Set<double[]>> e : bmus.entrySet())
			dBmus.put(grid.getPrototypeAt(e.getKey()), e.getValue());
		return DataUtils.getMeanQuantizationError(dBmus, dist);
	}

	// not normalized by neurons or samples
	public static <T> double getQuantizationError(Grid<double[]> grid, BmuGetter<double[]> bmuGetter, Dist<double[]> d, List<double[]> samples) {
		Map<GridPos, Set<double[]>> mapping = getBmuMapping(samples, grid, bmuGetter);
		Map<double[], Set<double[]>> clusters = new HashMap<double[], Set<double[]>>();
		for (GridPos k : mapping.keySet())
			clusters.put(grid.getPrototypeAt(k), mapping.get(k));
		return DataUtils.getSumOfSquares(clusters, d);
	}

	// how often are first and second nearest units in input space neighbours in output space
	public static double getTopoError(Grid<double[]> grid, BmuGetter<double[]> bmuGetter, List<double[]> samples) {
		if (grid.size() <= 2)
			return 0;

		int nAdj = 0;

		for (double[] x : samples) {
			GridPos firstPos = bmuGetter.getBmuPos(x, grid);
			Set<GridPos> ign = new HashSet<GridPos>();
			ign.add(firstPos);
			GridPos secondPos = bmuGetter.getBmuPos(x, grid, ign);

			if (grid.dist(firstPos, secondPos) > 1)
				nAdj++;
		}
		return (double) nAdj / samples.size();
	}

	public static final int PEARSON_TYPE = 0, SPEARMAN_TYPE = 1;

	public static double getTopoCorrelation(List<double[]> samples, Grid<double[]> grid, BmuGetter<double[]> bmuGetter, Dist<double[]> dist, int type) {
		Map<double[], GridPos> map = new HashMap<double[], GridPos>();
		for (double[] d : samples)
			map.put(d, bmuGetter.getBmuPos(d, grid));

		int size = (samples.size() * (samples.size() - 1)) / 2;
		double[] gridDist = new double[size];
		double[] vectorDist = new double[size];
		int index = 0;
		for (int i = 0; i < samples.size() - 1; i++) {
			for (int j = i + 1; j < samples.size(); j++) {
				double[] v1 = samples.get(i);
				double[] v2 = samples.get(j);

				vectorDist[index] = dist.dist(v1, v2);
				gridDist[index] = grid.dist(map.get(v1), map.get(v2));
				index++;
			}
		}

		if (index != size)
			throw new RuntimeException("Error in array length caclulation in pearson correlation!");

		if (type == PEARSON_TYPE) {
			PearsonsCorrelation cor = new PearsonsCorrelation();
			return cor.correlation(gridDist, vectorDist);
		} else if (type == SPEARMAN_TYPE) {
			SpearmansCorrelation cor = new SpearmansCorrelation();
			return cor.correlation(gridDist, vectorDist);
		} else
			throw new RuntimeException("Unknown correlation-type! " + type);
	}

	public static double getKangasError(List<double[]> samples, Grid<double[]> grid, KangasBmuGetter<double[]> kbg) {
		double error = 0;
		for (double[] x : samples) {
			GridPos bmuA = kbg.getBmuAPos(x, grid, null);
			GridPos bmuB = kbg.getBmuBPos(x, grid, null, bmuA);
			error += Math.pow((double) grid.dist(bmuA, bmuB) / (kbg.getRadius()), 2);
		}
		return error / samples.size();
	}

	// overlayable for u-matrix
	/*
	 * public static void drawPaths(Map<String,List<double[]>> individuals, Grid2D grid, BmuGetter bmuGetter, FileOutputStream os) { int CELL_SIZE = 50; Random r = new Random();
	 * 
	 * BufferedImage bufImg = new BufferedImage( CELL_SIZE * (2 * grid.getXDim()-1 ), CELL_SIZE * (2 * grid.getYDim()-1), BufferedImage.TYPE_INT_RGB); Graphics2D g2 = bufImg.createGraphics(); g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON); g2.setStroke(new BasicStroke(1f)); g2.setFont(new Font("Arial", Font.PLAIN, 12)); g2.setBackground( Color.GRAY ); g2.fillRect(0, 0, bufImg.getWidth(), bufImg.getHeight() );
	 * 
	 * int i = 1; for( String name : individuals.keySet() ) { List<double[]> l = individuals.get(name); g2.setColor( getColor( (float)(i++)/individuals.size() ) ); int rX = r.nextInt(20); int rY = r.nextInt(20);
	 * 
	 * int[] from = null; int k = 1; for( double[] s : l ) { int[] to = grid.getPosition( bmuGetter.getBmu( s, grid ) ); if( from != null ) { g2.fillOval(2*from[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*from[1]*CELL_SIZE+CELL_SIZE/2+rY, 5, 5); g2.fillOval(2*to[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*to[1]*CELL_SIZE+CELL_SIZE/2+rY, 5, 5); g2.drawLine(2*from[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*from[1]*CELL_SIZE+CELL_SIZE/2+rY, 2*to[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*to[1]*CELL_SIZE+CELL_SIZE/2+rY ); g2.drawString(""+(k++), 2*to[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*to[1]*CELL_SIZE+CELL_SIZE/2+rY ); } else { g2.drawString(name, 2*to[0]*CELL_SIZE+CELL_SIZE/2+rX, 2*to[1]*CELL_SIZE+CELL_SIZE/2+rY ); } from = to; } } g2.dispose();
	 * 
	 * try { ImageIO.write(bufImg, "png", os); } catch (IOException ex) { ex.printStackTrace(); } }
	 */

	public static void drawPaths(Map<String, List<double[]>> individuals, Grid2DHex<double[]> grid, BmuGetter<double[]> bmuGetter, FileOutputStream os) {
		int CELL_SIZE = 50;
		Random r = new Random();

		BufferedImage bufImg = new BufferedImage(CELL_SIZE * grid.getSizeOfDim(0), CELL_SIZE * grid.getSizeOfDim(1), BufferedImage.TYPE_INT_RGB);
		Graphics2D g2 = bufImg.createGraphics();
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.setStroke(new BasicStroke(2f));
		g2.setFont(new Font("Arial", Font.PLAIN, 24));
		g2.setBackground(Color.GRAY);
		g2.fillRect(0, 0, bufImg.getWidth(), bufImg.getHeight());

		int i = 1;
		for (String name : individuals.keySet()) {

			List<double[]> l = individuals.get(name);
			g2.setColor(Drawer.getColor((float) (i++) / individuals.size()));
			// g2.setColor( Color.BLACK );

			int rX = r.nextInt(10) - 5;
			int rY = r.nextInt(10) - 5;

			GridPos from = null;
			int k = 1;
			for (double[] s : l) {
				GridPos to = bmuGetter.getBmuPos(s, grid);
				if (from != null) {
					int radius = 5;
					g2.fillOval(from.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX - radius / 2, from.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY - radius / 2, radius, radius);
					g2.fillOval(to.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX - radius / 2, to.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY - radius / 2, radius, radius);

					g2.drawLine(from.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX, from.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY, to.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX, to.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY);
					g2.drawString("" + (k++), to.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX, to.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY);
				} else {
					g2.drawString(name, to.getPos(0) * CELL_SIZE + CELL_SIZE / 2 + rX, to.getPos(1) * CELL_SIZE + CELL_SIZE / 2 + rY);
				}
				from = to;
			}
		}
		g2.dispose();

		try {
			ImageIO.write(bufImg, "png", os);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	public static void saveGrid(Grid<double[]> grid, OutputStream os) {
		int vDim = grid.getPrototypes().iterator().next().length;

		Element root = new Element("grid");
		root.setAttribute("hex", (grid instanceof Grid2DHex<?>) + "");
		root.setAttribute("toroid", (grid instanceof Grid2DHexToroid<?> || grid instanceof Grid2DToroid<?>) + "");
		Element u = new Element("units");

		for (GridPos pos : grid.getPositions()) {
			Element p = new Element("unit");
			p.setAttribute("x", pos.getPos(0) + "");
			p.setAttribute("y", pos.getPos(1) + "");

			Element e = new Element("vector");
			double[] vec = grid.getPrototypeAt(pos);
			for (int i = 0; i < vDim; i++) {
				Element v = new Element("value");
				v.setText(vec[i] + "");
				e.addContent(v);
			}
			p.addContent(e);
			u.addContent(p);
		}
		root.addContent(u);

		Document doc = new Document(root);
		try {
			XMLOutputter serializer = new XMLOutputter();
			serializer.setFormat(Format.getPrettyFormat());
			serializer.output(doc, os);
		} catch (IOException e) {
			System.err.println(e);
		}
	}

	public static Grid2D<double[]> loadGrid(InputStream is) {
		HashMap<GridPos, double[]> map = null;
		boolean toroid = false, hex = false;

		try {
			SAXBuilder builder = new SAXBuilder();
			Document doc = builder.build(is);
			Element root = doc.getRootElement();
			map = new HashMap<GridPos, double[]>();

			hex = Boolean.parseBoolean(root.getAttribute("hex").getValue());
			toroid = Boolean.parseBoolean(root.getAttribute("toroid").getValue());

			for (Object o1 : root.getChild("units").getChildren()) {
				Element e = (Element) o1;

				GridPos p = new GridPos(new int[] { Integer.parseInt(e.getAttribute("x").getValue()), Integer.parseInt(e.getAttribute("y").getValue()) });

				List<Double> v = new ArrayList<Double>();
				for (Object o2 : e.getChild("vector").getChildren()) {
					Element e2 = (Element) o2;
					v.add(Double.parseDouble(e2.getText()));
				}

				double[] va = new double[v.size()];
				for (int i = 0; i < va.length; i++)
					va[i] = v.get(i);

				map.put(p, va);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		if (hex) {
			if (toroid)
				return new Grid2DHexToroid<>(map);
			else
				return new Grid2DHex<>(map);
		} else {
			if (toroid)
				return new Grid2DToroid<>(map);
			else
				return new Grid2D<>(map);
		}
	}

	public static void initRandom(Grid<double[]> grid, List<double[]> samples) {
		Random r = new Random(0);
		for (GridPos p : grid.getPositions()) {
			double[] d = samples.get(r.nextInt(samples.size()));
			grid.setPrototypeAt(p, Arrays.copyOf(d, d.length));
		}
	}

	//TODO buggy and not correct, presumably
	/*public static void initLinear(Grid2D<double[]> grid, List<double[]> samples, boolean scaled) {
		RealMatrix matrix = new Array2DRowRealMatrix(samples.size(), samples.get(0).length);
		for (int i = 0; i < samples.size(); i++)
			matrix.setRow(i, samples.get(i));

		SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
		// RealMatrix v = svd.getV().multiply( svd.getS().scalarMultiply(1.0/Math.sqrt(samples.size()-1))); // loadings
		RealMatrix v = svd.getV(); // Eigenvectors

		double[] firstComponent = v.getRow(0);
		double[] secondComponent = v.getRow(1);

		// get mean
		BasicStatistics bs = new BasicStatistics();
		double[] m = bs.meanVector(s);

		int xDim = grid.getSizeOfDim(0);
		int yDim = grid.getSizeOfDim(1);

		for (int i = 0; i < xDim; i++) {
			for (int j = 0; j < yDim; j++) {
				double a = (double) i / xDim - 0.5;
				double b = (double) j / yDim - 0.5;
				double[] d = new double[vLength];
				for (int k = 0; k < vLength; k++)
					d[k] = m[k] + a * firstComponent[k] + b * secondComponent[k];
				grid.setPrototypeAt(new GridPos(i, j), d);
			}
		}
	}*/

	public static <T> void printGeoGrid(int[] ga, Grid<double[]> grid, String fn) {
		try {
			printGeoGrid(ga, grid, new FileOutputStream(fn));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static <T> void printGeoGrid(int[] ga, Grid<double[]> grid, OutputStream os) {
		Map<GridPos, Coordinate> map = new HashMap<GridPos, Coordinate>();

		GeometryFactory gf = new GeometryFactory();
		for (GridPos p : grid.getPositions())
			map.put(p, new Coordinate(grid.getPrototypeAt(p)[ga[0]], grid.getPrototypeAt(p)[ga[1]]));

		SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
		typeBuilder.setName("topo");
		typeBuilder.add("the_geom", LineString.class);

		SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(typeBuilder.buildFeatureType());
		DefaultFeatureCollection fc = new DefaultFeatureCollection();

		for (GridPos p : map.keySet()) {
			for (GridPos nb : grid.getNeighbours(p)) {
				LineString ls = gf.createLineString(new Coordinate[] { map.get(p), map.get(nb) });
				featureBuilder.set("the_geom", ls);
				fc.add(featureBuilder.buildFeature("" + fc.size()));
			}
		}

		try {
			StyleBuilder sb = new StyleBuilder();
			MapContent mc = new MapContent();
			mc.addLayer(new FeatureLayer(fc, SLD.wrapSymbolizers(sb.createLineSymbolizer(Color.BLACK, 2.0))));

			GTRenderer renderer = new StreamingRenderer();
			renderer.setMapContent(mc);

			Rectangle imageBounds = null;

			ReferencedEnvelope mapBounds = mc.getMaxBounds();
			double heightToWidth = mapBounds.getSpan(1) / mapBounds.getSpan(0);
			int imageWidth = 2000;
			imageBounds = new Rectangle(0, 0, imageWidth, (int) Math.round(imageWidth * heightToWidth));

			BufferedImage image = new BufferedImage(imageBounds.width, imageBounds.height, BufferedImage.TYPE_INT_ARGB);
			Graphics2D gr = image.createGraphics();
			// gr.setPaint(Color.WHITE);
			// gr.fill(imageBounds);
			renderer.paint(gr, imageBounds, mapBounds);

			ImageIO.write(image, "png", os);
			image.flush();
			mc.dispose();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
