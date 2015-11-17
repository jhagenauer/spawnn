package spawnn.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.random.GaussianRandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.triangulate.VoronoiDiagramBuilder;

public class SampleBuilder {

	private static Logger log = Logger.getLogger(SampleBuilder.class);

	public static void buildRandomSamples(int numSamples, int numAttrs, String fn) {
		Random r = new Random();

		Envelope env = new Envelope(0, 1, 0, 1);
		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (int i = 0; i < numSamples; i++)
			coords.add(new Coordinate(r.nextDouble(), r.nextDouble()));
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> geoms = new ArrayList<Geometry>();
		List<double[]> samples = new ArrayList<double[]>();

		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));

			Point p = geom.getCentroid();

			geoms.add(geom);
			double[] d = new double[2 + numAttrs];
			d[0] = p.getX();
			d[1] = p.getY();
			for (int j = 0; j < numAttrs; j++)
				d[2 + j] = r.nextDouble();
			samples.add(d);
		}

		List<String> names = new ArrayList<String>();
		names.add("X");
		names.add("Y");

		for (int i = 2; i < samples.get(0).length; i++)
			names.add("data" + i);

		DataUtils.writeShape(samples, geoms, names.toArray(new String[names.size()]), fn);
	}

	public static void buildRandomRegions(int numCluster, int numSamples, double noise, int maxColors, String fn) {
		final Random r = new Random();

		Envelope env = new Envelope(0, 1, 0, 1);
		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (int i = 0; i < numSamples; i++)
			coords.add(new Coordinate(r.nextDouble(), r.nextDouble()));
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> geoms = new ArrayList<Geometry>();
		List<double[]> samples = new ArrayList<double[]>();

		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));

			Point p = geom.getCentroid();

			geoms.add(geom);
			double[] d = new double[] { p.getX(), p.getY(), -1 };
			samples.add(d);
		}

		Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();
		for (int i = 0; i < numCluster; i++)
			classes.put(i, new HashSet<double[]>());

		// seed
		int assigned = 0;
		for (int i = 0; i < numCluster; i++) {
			double[] d = null;
			do {
				d = samples.get(r.nextInt(numSamples));
			} while (d[2] >= 0);
			d[2] = i;
			classes.get(i).add(d);
			assigned++;
		}

		// grow cluster
		while (assigned < numSamples) {

			// "randomly" get a class and sample
			int c = r.nextInt(numCluster);

			double[] d = null;
			while (d == null)
				for (double[] t : classes.get(c))
					if (r.nextDouble() < 1.0 / classes.get(c).size()) {
						d = t;
						break;
					}

			Geometry geom = geoms.get(samples.indexOf(d));

			// check surrounding samples for free one
			List<double[]> candidates = new ArrayList<double[]>();
			for (int i = 0; i < samples.size(); i++) {
				if (geoms.get(i).touches(geom) && samples.get(i)[2] < 0)
					candidates.add(samples.get(i));
			}

			if (candidates.size() > 0) {
				double[] d2 = candidates.get(r.nextInt(candidates.size()));
				d2[2] = c;
				classes.get(c).add(d2);
				assigned++;
			}
		}

		Set<Double> colors;
		do {
			log.debug("greedy coloring");
			// reset values
			for (double[] d : samples)
				d[2] = 0;

			List<Integer> cs = new ArrayList<Integer>(classes.keySet());
			Collections.shuffle(cs);

			for (int c : cs) {
				Set<double[]> l = classes.get(c);

				// get neighbors
				List<double[]> nbs = new ArrayList<double[]>();
				for (double[] d : l) {
					int idx = samples.indexOf(d);
					for (int j = 0; j < samples.size(); j++) {
						if (geoms.get(j).touches(geoms.get(idx)) && !l.contains(samples.get(j)))
							nbs.add(samples.get(j));
					}
				}

				// get neighboring colors
				Set<Integer> nbC = new HashSet<Integer>();
				for (double[] d : nbs)
					for (int nc : classes.keySet()) {
						if (classes.get(nc).contains(d))
							nbC.add((int) d[2]);
					}

				// use lowest not used color
				int color = 0;
				for (; nbC.contains(color); color++)
					;

				// assign color
				for (double[] d : l)
					d[2] = color;
			}

			colors = new HashSet<Double>();
			for (double[] d : samples)
				colors.add(d[2]);
			log.debug("Colors: " + colors.size());
		} while (colors.size() > maxColors);

		// add some noise
		for (double[] d : samples)
			d[2] += noise * r.nextDouble() - noise / 2;

		// add classes
		List<double[]> samplesWithClass = new ArrayList<double[]>();
		for (double[] d : samples) {
			double[] nd = Arrays.copyOf(d, d.length + 1);
			for (int i : classes.keySet())
				if (classes.get(i).contains(d))
					nd[nd.length - 1] = i;
			samplesWithClass.add(nd);
		}

		DataUtils.writeShape(samplesWithClass, geoms, new String[] { "X", "Y", "VALUE", "CLASS" }, fn);
	}

	public static void buildConcentricCircles(int numSamples, int numCircles, String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		/*
		 * Envelope env = new Envelope(0, 1, 0, 1); List<Coordinate> coords = new ArrayList<Coordinate>(); for (int i = 0; i < numSamples; i++) coords.add(new Coordinate(r.nextDouble(), r.nextDouble())); VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder(); vdb.setSites(coords); GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);
		 */

		List<double[]> samples = new ArrayList<double[]>();
		Point center = gf.createPoint(new Coordinate(0.5, 0.5));

		while (samples.size() < numSamples) {
			Point p = gf.createPoint(new Coordinate(r.nextDouble(), r.nextDouble()));

			double dist = center.distance(p);
			int val = -1;
			int cl = -1;
			for (double j = 0; j < numCircles; j++) {
				if (dist > 0.5 * j / numCircles && dist < 0.5 * (j + 1) / numCircles) {
					val = (int) j % 2;
					cl = (int) j;
				}
			}

			if (cl >= 0)
				samples.add(new double[] { p.getX(), p.getY(), val, cl });
		}

		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static List<double[]> readSamplesFromFcps(String lrn, String cls) {
		// read classes
		Map<Integer, Integer> classes = new HashMap<Integer, Integer>();
		BufferedReader r = null;
		try {
			r = new BufferedReader(new FileReader(cls));

			log.debug(r.readLine());

			String line = null;
			while ((line = r.readLine()) != null) {
				String[] s = line.split("\t");
				classes.put(Integer.parseInt(s[0]), Integer.parseInt(s[1]));
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				r.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		// read learn data/samples
		List<double[]> samples = new ArrayList<double[]>();
		int dim = -1;
		r = null;
		try {
			r = new BufferedReader(new FileReader(lrn));

			log.debug(r.readLine());
			dim = Integer.parseInt(r.readLine().split(" ")[1]) - 1;

			log.debug(r.readLine());
			log.debug(r.readLine());

			String line = null;
			while ((line = r.readLine()) != null) {

				double[] d = new double[dim + 1];
				String[] s = line.split("\t");

				int id = Integer.parseInt(s[0]);

				for (int i = 1; i < dim + 1; i++)
					d[i - 1] = Double.parseDouble(s[i]);
				// log.debug("id: " + id);
				d[dim] = classes.get(id); // class information

				samples.add(d);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				r.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return samples;
	}

	public static void buildNDiffDensSquares(int n, int numSamples, double ratio, double noise, double width, double height, String fn) {
		Random r = new Random();

		Envelope env = new Envelope(0, width, 0, height);

		List<double[]> samples = new ArrayList<double[]>();

		for (int i = 0; i < n; i++) {
			if (i % 2 == 0) {
				Set<double[]> s = new HashSet<double[]>();
				while (s.size() < ratio * numSamples / n) {
					double x = i * env.getWidth() / n + r.nextDouble() * env.getWidth() / n;
					double y = r.nextDouble() * env.getHeight();
					s.add(new double[] { x, y, 0, i });
				}
				samples.addAll(s);
			} else {
				Set<double[]> s = new HashSet<double[]>();
				while (s.size() < (1 - ratio) * numSamples / n) {
					double x = i * env.getWidth() / n + r.nextDouble() * env.getWidth() / n;
					double y = r.nextDouble() * env.getHeight();
					s.add(new double[] { x, y, 1 - noise / 2 + r.nextDouble() * noise, i });
				}
				samples.addAll(s);
			}
		}

		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (double[] d : samples)
			coords.add(new Coordinate(d[0], d[1]));

		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> geoms = new ArrayList<Geometry>();
		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));
			geoms.add(geom);
		}

		// sort
		List<Geometry> ng = new ArrayList<Geometry>();
		for (Coordinate c : coords) {
			Point p = gf.createPoint(c);
			for (Geometry g : geoms)
				if (g.contains(p))
					ng.add(g);
		}
		DataUtils.writeShape(samples, ng, new String[] { "X", "Y", "Value", "Cluster" }, fn);
	}

	// TODO: adjust denistiy and vairance to give best results (best: spu
	// 0.0005, dens = 0.1, var = 0.5 )
	public static void buildRandomDiffDensRegions(int numCluster, String fn) {
		final Random r = new Random();

		Map<Integer, Set<double[]>> classes = new HashMap<Integer, Set<double[]>>();

		Envelope env = new Envelope(0, 1, 0, 1);
		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (int i = 0; i < numCluster; i++)
			coords.add(new Coordinate(r.nextDouble(), r.nextDouble()));

		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> clusterGeoms = new ArrayList<Geometry>();
		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));
			clusterGeoms.add(geom);
		}

		log.debug("greedy coloring");
		Set<Integer> usedColors;
		Map<Geometry, Integer> coloring;
		do {

			coloring = new HashMap<Geometry, Integer>();
			// reset values
			for (Geometry g : clusterGeoms)
				coloring.put(g, 0);

			List<Geometry> cs = new ArrayList<Geometry>(clusterGeoms);
			Collections.shuffle(cs);

			for (Geometry c : cs) {

				// get colors of neighboring clusterGeoms
				Set<Integer> nbC = new HashSet<Integer>();
				for (Geometry g : clusterGeoms)
					if (g.touches(c) && !c.equals(g))
						nbC.add(coloring.get(g));

				// use lowest not used color of nbs
				int color = 0;
				for (; nbC.contains(color); color++)
					;

				// assign color
				coloring.put(c, color);
			}

			usedColors = new HashSet<Integer>(coloring.values());
			log.debug("Colors: " + usedColors.size());

		} while (usedColors.size() > 4);

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();

		for (Geometry geom : clusterGeoms) {

			int cl = clusterGeoms.indexOf(geom);
			int color = coloring.get(geom);
			double samplesPerUnit = 0.0005; // = 100% feature per area unit
			double area = geom.getArea() / env.getArea();

			// 4 colors,
			Set<Coordinate> s = new HashSet<Coordinate>();
			if (color == 0 || color == 1) {

				while (s.size() < 0.1 * area / samplesPerUnit) { // %10 density
					double x = r.nextDouble();
					double y = r.nextDouble();
					Coordinate c = new Coordinate(x, y);

					if (gf.createPoint(c).within(geom))
						s.add(c);
				}
			} else {
				while (s.size() < area / samplesPerUnit) { // %100 density
					double x = r.nextDouble();
					double y = r.nextDouble();
					Coordinate c = new Coordinate(x, y);

					Point p = gf.createPoint(c);
					if (p.within(geom))
						s.add(c);
				}
			}

			// build voro + samples
			VoronoiDiagramBuilder vdb2 = new VoronoiDiagramBuilder();
			vdb2.setClipEnvelope(env);
			vdb2.setSites(s);

			GeometryCollection coll2 = (GeometryCollection) vdb2.getDiagram(gf);
			for (int j = 0; j < coll2.getNumGeometries(); j++) {

				Geometry geom2 = coll2.getGeometryN(j).intersection(geom);

				Point p = geom2.getCentroid();

				double v = color;
				if (color == 0 | color == 2) // high noise/variance, else no
												// noise
					v += r.nextDouble() * 0.5;

				double[] d = new double[] { p.getX(), p.getY(), v, cl };
				samples.add(d);
				geoms.add(geom2);

				if (!classes.containsKey(cl))
					classes.put(cl, new HashSet<double[]>());

				classes.get(cl).add(d);
			}
		}

		int smallestClusterSize = Integer.MAX_VALUE;
		for (Set<double[]> s : classes.values())
			if (s.size() < smallestClusterSize)
				smallestClusterSize = s.size();
		log.debug("smallest cluster: " + smallestClusterSize);

		log.debug("samples: " + samples.size());

		try {
			Drawer.geoDrawCluster(classes.values(), samples, geoms, new FileOutputStream("output/test.png"), true);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		DataUtils.writeShape(samples, geoms, fn);
	}

	public static void buildSPSDataSet(int numSamples, double spaceDist, double timeDist, String out) {
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();

		while (samples.size() < numSamples) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z = r.nextDouble();

			double s = -1;
			int c = -1; // cluster
			if (x < 0.5 - spaceDist && z < 0.5 - timeDist) {
				s = 0;
				c = 1;
			} else if (x > 0.5 + spaceDist && z < 0.5 - timeDist) {
				s = 0;
				c = 2;
			} else if (x < 0.5 - spaceDist && z > 0.5 + timeDist) {
				s = 0;
				c = 3;
			} else if (x > 0.5 + spaceDist && z > 0.5 + timeDist) {
				s = 0;
				c = 4;
			} else {
				s = 1;
				c = 0;
			}

			double[] d = new double[] { x, y, z, s, c };
			samples.add(d);
		}
		try {
			DataUtils.writeCSV(new FileOutputStream(out), samples);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public static void buildAustriaSet() {
		List<double[]> origSamples = DataUtils.readSamplesFromShapeFile(new File("data/sps/municipalitiesAustria/trend.shp"), new int[] {}, true);
		List<Geometry> origGeoms = DataUtils.readGeometriesFromShapeFile(new File("data/sps/municipalitiesAustria/trend.shp"));

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> geoms = new ArrayList<Geometry>();

		for (int i = 0; i < origSamples.size(); i++) {

			double[] d = origSamples.get(i);
			Geometry g = origGeoms.get(i);

			double area = g.getArea();

			// 61
			samples.add(new double[] { 1961, d[13], d[14], d[16], d[29] / d[16], d[42] / d[16], d[16] / area, d[88] / area });
			geoms.add(g);

			// 71
			samples.add(new double[] { 1971, d[13], d[14], d[19], d[32] / d[19], d[45] / d[19], d[19] / area, d[91] / area });
			geoms.add(g);

			// 81
			samples.add(new double[] { 1981, d[13], d[14], d[22], d[35] / d[22], d[48] / d[22], d[22] / area, d[94] / area });
			geoms.add(g);

			// 91
			samples.add(new double[] { 1991, d[13], d[14], d[25], d[38] / d[25], d[51] / d[25], d[25] / area, d[97] / area });
			geoms.add(g);

			// 01
			samples.add(new double[] { 2001, d[13], d[14], d[28], d[41] / d[28], d[54] / d[28], d[28] / area, d[100] / area });
			geoms.add(g);
		}

		// Bevlk = Bevölkerung, RBschft = % an Beschäftigten, RAspndl = % an Auspendlerm, Bvlkdcht = Bevölkerungsdichte, NLBtdcht = Dichte nichtlandwirtschaftlicher Betriebe
		DataUtils.writeShape(samples, geoms, new String[] { "Year", "X", "Y", "Bvlk", "RBschft", "RAspndl", "Bvlkdcht", "NLBtdcht" }, "data/sps/munaus.shp");
	}

	public static void buildArtifialSpatialDataSet(int numSamples, String fn) {
		Random r = new Random();

		Envelope env = new Envelope(0, 1, 0, 1);
		List<Coordinate> coords = new ArrayList<Coordinate>();
		for (int i = 0; i < numSamples; i++)
			coords.add(new Coordinate(r.nextDouble(), r.nextDouble()));
		VoronoiDiagramBuilder vdb = new VoronoiDiagramBuilder();
		vdb.setSites(coords);

		GeometryFactory gf = new GeometryFactory();
		GeometryCollection coll = (GeometryCollection) vdb.getDiagram(gf);

		List<Geometry> geoms = new ArrayList<Geometry>();
		List<double[]> samples = new ArrayList<double[]>();

		for (int i = 0; i < coll.getNumGeometries(); i++) {
			Geometry geom = coll.getGeometryN(i);
			geom = geom.intersection(gf.toGeometry(env));

			Point p = geom.getCentroid();

			geoms.add(geom);
			// geoms.add(p);
			double[] d = new double[] { p.getX(), p.getY(), r.nextDouble() };
			//double[] d = new double[] { p.getX(), p.getY(), Math.pow(p.getX() - p.getY(), 2) };
			samples.add(d);
		}
		DataUtils.writeShape(samples, geoms, new String[] { "X", "Y", "VALUE" }, fn);
	}

	public static void buildNSquares(double[] widths, double[] noise, int numSamples, double height, boolean eqProb, String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		double width = 0;
		for (double w : widths)
			width += w;

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		// get cluster to add
		while (samples.size() < numSamples) {
			int c = 1;
			if (eqProb) {
				c = r.nextInt(widths.length);
			} else {
				double d = r.nextDouble() * width;
				double a = 0;
				for (int i = 0; i < widths.length; i++) {
					if (a < d && d < a + widths[i]) {
						c = i;
						break;
					}
					a += widths[i];
				}
			}

			// get lower cluster limit
			double a = 0;
			for (int i = 0; i < c; i++) {
				a += widths[i];
			}

			double x = a += r.nextDouble() * widths[c];
			double y = r.nextDouble() * height;
			double v = (c % 2 == 0) ? 1 : 0;

			// add noise
			v = v - noise[c] / 2.0 + r.nextDouble() * noise[c];

			samples.add(new double[] { x, y, v, c });
			points.add(gf.createPoint(new Coordinate(x, y)));

		}

		DataUtils.writeShape(samples, points, new String[] { "X", "Y", "Value", "Cluster" }, fn);
	}

	public static void buildDiamond(int numSamples, String fn, boolean eqClDens) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		// get cluster to add
		while (samples.size() < numSamples) {

			int tc = r.nextInt(5);

			double x, y, v, c;
			do {

				x = r.nextDouble();
				y = r.nextDouble();
				v = 0;
				c = 0;

				if (x + y < 0.5) {
					v = 1;
					c = 1;
				} else if (y + (1 - x) < 0.5) {
					v = 1;
					c = 2;
				} else if ((1 - y) + x < 0.5) {
					v = 1;
					c = 3;
				} else if ((1 - y) + (1 - x) < 0.5) {
					v = 1;
					c = 4;
				}

			} while (eqClDens && c != tc);

			samples.add(new double[] { x, y, v, c });
			points.add(gf.createPoint(new Coordinate(x, y)));

		}
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
		// DataUtils.writeToShape(samples, points, new String[]{"X","Y","Value","Cluster"}, fn);
	}

	public static void buildLTest(String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		for (int i = 0; i < 1000; i++) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double v;
			double c;

			if (x > 3d / 4) {
				v = 1;
				c = 2;
			} else if (x > 2d / 4) {
				v = 0;
				c = 1;
			} else if (x < 1d / 4) {
				v = 1;
				c = 0;
			} else if (y > .5) {
				v = 0;
				c = 1;
			} else {
				v = 1;
				c = 0;
			}

			x *= 2;

			samples.add(new double[] { x, y, v, c });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}
		// DataUtils.writeToShape(samples, points, new String[]{"X","Y","Value","Cluster"}, fn);
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "cluster" });
	}

	public static void buildTTest(String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		for (int i = 0; i < 800; i++) {
			double x = r.nextDouble() * 6;
			double y = r.nextDouble() * 5;
			double v;
			double c;

			if (x < 1 || (x < 4 && y < 1) || (x < 4 && y > 4)) {
				v = 1;
				c = 0;
			} else if (x > 5 || (x > 2 && y > 2 && y < 3)) {
				v = 1;
				c = 2;
			} else {
				v = 0;
				c = 1;
			}

			samples.add(new double[] { x, y, v, c });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		// DataUtils.writeToShape(samples, points, new String[]{"X","Y","Value","Cluster"}, fn);
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildSquareTest2(String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		for (int i = 0; i < 500; i++) {
			double x = r.nextDouble() * 0.5;
			double y = r.nextDouble();
			double v = 0;
			samples.add(new double[] { x, y, v, 0 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		for (int i = 0; i < 500; i++) {
			double x = r.nextDouble() * 0.5 + 0.5;
			double y = r.nextDouble();

			double v = r.nextDouble();
			samples.add(new double[] { x, y, v, 1 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}
		DataUtils.writeShape(samples, points, new String[] { "X", "Y", "Value", "Cluster" }, fn);
	}

	public static void buildSquareTest3(String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		// min spatial var, max attribte var
		for (int i = 0; i < 500; i++) {
			double x = 0.5;
			double y = 0.5;
			double v = r.nextDouble();
			samples.add(new double[] { x, y, v, 0 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		// max spatial var, min attribte var
		for (int i = 0; i < 500; i++) {
			double x = r.nextDouble() + 1;
			double y = r.nextDouble();
			double v = 0.5;
			samples.add(new double[] { x, y, v, 1 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		// min spatial var, min attirbute var
		for (int i = 0; i < 500; i++) {
			double x = 0.5 + 1;
			double y = 0.5 + 1;
			double v = 0.5;
			samples.add(new double[] { x, y, v, 2 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		// max spatial var, max attirbute var
		for (int i = 0; i < 500; i++) {
			double x = r.nextDouble();
			double y = r.nextDouble() + 1;
			double v = r.nextDouble();
			samples.add(new double[] { x, y, v, 3 });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}

		DataUtils.writeShape(samples, points, new String[] { "X", "Y", "Value", "Cluster" }, fn);
	}

	public static void buildSyntheticSurface(int num, String fn) {
		List<double[]> samples = new ArrayList<double[]>();

		double pointsPerDim = Math.sqrt(num);

		for (double x = -2; x <= 2; x += 4.0 / pointsPerDim) {
			for (double y = -2; y <= 2; y += 4.0 / pointsPerDim) {

				x = (double) Math.round(x * 10000) / 10000;
				y = (double) Math.round(y * 10000) / 10000;

				double z = Math.pow(x, 2) * Math.exp(-Math.pow(x, 2) - Math.pow(y + 1, 2));
				z -= (0.2 * x - Math.pow(x, 3) - Math.pow(y, 5)) * Math.exp(-Math.pow(x, 2) - Math.pow(y, 2));

				samples.add(new double[] { x, y, z });
			}
		}
		log.debug(samples.size());
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "z" });
	}

	public static void buildSyntheticSurfaceR(int num, String fn) {
		JDKRandomGenerator r = new JDKRandomGenerator();
		GaussianRandomGenerator grg = new GaussianRandomGenerator(r);
		List<double[]> samples = new ArrayList<double[]>();

		while (samples.size() != num) {
			double x = r.nextDouble() * 4 - 2;
			double y = r.nextDouble() * 4 - 2;

			double z = Math.pow(x, 2) * Math.exp(-Math.pow(x, 2) - Math.pow(y + 1, 2));
			z -= (0.2 * x - Math.pow(x, 3) - Math.pow(y, 5)) * Math.exp(-Math.pow(x, 2) - Math.pow(y, 2));
			z += z*grg.nextNormalizedDouble();

			samples.add(new double[] { x, y, z });
		}
		log.debug(samples.size());
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "z" });
	}

	// TODO
	public static void buildGaussianMixture(int numSamples, int nCluster, String fn) {
		Random r = new Random();
		GeometryFactory gf = new GeometryFactory();

		List<double[]> samples = new ArrayList<double[]>();
		List<Geometry> points = new ArrayList<Geometry>();

		MultivariateNormalDistribution[] mnvs = new MultivariateNormalDistribution[nCluster];
		for (int i = 0; i < nCluster; i++) {
			double meanX = r.nextDouble() - 0.5;
			double meanY = r.nextDouble() - 0.5;

			log.debug(meanX + "," + meanY);
			mnvs[i] = new MultivariateNormalDistribution(new double[] { meanX, meanY }, new double[][] { { 1, r.nextDouble() }, { r.nextDouble(), 1 } });
		}

		while (samples.size() < numSamples) {
			double x = r.nextDouble() * 4 - 2;
			double y = r.nextDouble() * 4 - 2;
			double v = 0;

			for (MultivariateNormalDistribution mnv : mnvs) {
				// v += mnv.density( new double[]{ x, y } );
				v = Math.max(v, mnv.density(new double[] { x, y }));
			}

			samples.add(new double[] { x, y, v });
			points.add(gf.createPoint(new Coordinate(x, y)));
		}
		DataUtils.writeShape(samples, points, new String[] { "X", "Y", "Value" }, fn);
	}

	public static void buildGaussianMixture( int numSamples, String fn) {
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();

		List<MultivariateNormalDistribution> mnvs = new ArrayList<MultivariateNormalDistribution>();
		mnvs.add(new MultivariateNormalDistribution(new double[] { -0.3, 0 }, new double[][] { { 1.0, 0.1 }, { 0.2, 0.9 } }));
		mnvs.add(new MultivariateNormalDistribution(new double[] { 1.2, 2 }, new double[][] { { 0.8, -0.4 }, { -0.4, 0.7 } }));

		// mnvs.add( new MultivariateNormalDistribution( new double[]{ 0, 0 } , new double[][]{{1, 0.0 },{0.0, 1 }} ) );
		// mnvs.add( new MultivariateNormalDistribution( new double[]{ 1.5, 1.5 } , new double[][]{{1, -0.9 },{0.8, 1 }} ) );

		Dist<double[]> dist = new EuclideanDist();
		for (int i = 0; i < mnvs.size(); i++) {
			MultivariateNormalDistribution mnv = mnvs.get(i);

			while (samples.size() / (i + 1) < numSamples) {
				double x = r.nextDouble() * 40 - 20;
				double y = r.nextDouble() * 40 - 20;

				double[] p = new double[] { x, y };
				if (mnv.density(p) > r.nextDouble()) {

					samples.add(new double[] { x, y, -dist.dist(mnv.getMeans(), p), i });
				}

			}

		}
		DataUtils.normalizeColumn(samples, 2);
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildTwoPlanes(String fn) {
		List<double[]> samples = new ArrayList<double[]>();
		for (double x = -1; x <= 1; x += 0.05)
			for (double y = 0; y <= 1; y += 0.05)
				if (x < 0)
					samples.add(new double[] { x, y, -1 });
				else
					samples.add(new double[] { x, y, 1 });

		log.debug(samples.size());
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "z" });
	}

	public static void buildSquareville(String fn) {
		List<double[]> samples = new ArrayList<double[]>();
		for (double x = 0; x <= 1.00001; x += 0.1)
			for (double y = 0; y <= 2.00001; y += 0.1) {
				if (y <= 0.5)
					samples.add(new double[] { x, y, 10, 0 });
				else if (y >= 1.5)
					samples.add(new double[] { x, y, 10, 2 });
				else
					samples.add(new double[] { x, y, 0, 1 });
			}
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildSquarevilleX(String fn) {
		List<double[]> samples = new ArrayList<double[]>();

		for (double y = 0; y <= 2.00001; y += 0.1) {
			for (double x = 0; x <= 1.00001; x += 0.1) {

				if (y <= 0.5)
					samples.add(new double[] { x, y, x, 0 });
				else if (y >= 1.5)
					samples.add(new double[] { x, y, x, 2 });
				else
					samples.add(new double[] { x, y, 1 - x, 1 });

			}
		}
		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildOutlier(String fn) {
		Random r = new Random();
		List<double[]> samples = new ArrayList<double[]>();

		while (samples.size() < 400) {
			double x = r.nextDouble() * 2;
			double y = r.nextDouble() * 2;

			if (2 - x + y < 2)
				samples.add(new double[] { x - 1, y - 1, 1, 0 });
			else
				samples.add(new double[] { x - 1, y - 1, -1, 1 });
		}

		// "spatial outlier"
		for (int i = 0; i < 5; i++)
			samples.add(new double[] { r.nextDouble() + 10, r.nextDouble() + 10, 1, 0 });
		for (int i = 0; i < 5; i++)
			samples.add(new double[] { r.nextDouble() - 10, r.nextDouble() - 10, -1, 1 });

		// feature outlier
		int oc = 0;
		while (oc < 5) {
			double x = r.nextDouble() * 2;
			double y = r.nextDouble() * 2;

			if (2 - x + y < 2) {
				samples.add(new double[] { x - 1, y - 1, -10, 0 });
				oc++;
			}
		}

		oc = 0;
		while (oc < 5) {
			double x = r.nextDouble() * 2;
			double y = r.nextDouble() * 2;

			if (2 - x + y > 2) {
				samples.add(new double[] { x - 1, y - 1, +10, 1 });
				oc++;
			}
		}

		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void build4Corners(String fn) {
		List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : DataUtils.readCSV("data/4corners.csv")) {
			double x = d[0];
			double y = d[1];

			double[] nd = Arrays.copyOf(d, d.length + 1);
			int cc = nd.length - 1;

			if (x < 5 && y < 5)
				nd[cc] = 1;
			else if (x > 15 && y < 5)
				nd[cc] = 2;
			else if (x < 5 && y > 10)
				nd[cc] = 3;
			else if (x > 15 && y > 10)
				nd[cc] = 4;
			else
				nd[cc] = 0;
			samples.add(nd);
		}

		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildNoisyCorners(String fn) {
		Random r = new Random();

		List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : DataUtils.readCSV("data/4corners.csv")) {
			double x = d[0];
			double y = d[1];

			double[] nd = Arrays.copyOf(d, d.length + 1);
			int c = nd.length - 2;
			int cc = nd.length - 1;

			nd[c] = 0;
			nd[cc] = 0;
			if (x < 5 && y < 5) {
				nd[cc] = 1;
				nd[c] = 1;
			} else if (x > 15 && y < 5) {
				if (r.nextDouble() < 0.5)
					nd[c] = 1;
				nd[cc] = 2;
			} else if (x < 5 && y > 10) {
				if (r.nextDouble() < 0.25)
					nd[c] = 1;
				nd[cc] = 3;
			} else if (x > 15 && y > 10) {
				if (r.nextDouble() < 0.1)
					nd[c] = 1;
				nd[cc] = 4;
			}
			samples.add(nd);
		}

		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void buildVarCorners(String fn) {
		List<double[]> samples = new ArrayList<double[]>();
		for (double[] d : DataUtils.readCSV("data/4corners.csv")) {

			double[] nd = Arrays.copyOf(d, d.length + 1);
			int c = nd.length - 2;
			int cc = nd.length - 1;

			double x = d[0];
			double y = d[1];
			if (x < 5 && y < 5) {
				nd[c] = 1;
				nd[cc] = 1;
			} else if (x > 15 && y < 5) {
				nd[c] = 0.5;
				nd[cc] = 2;
			} else if (x < 5 && y > 10) {
				nd[c] = 0.1;
				nd[cc] = 3;
			} else if (x > 15 && y > 10) {
				nd[c] = 1;
				nd[cc] = 4;
			} else {
				nd[c] = 0;
				nd[cc] = 0;
			}
			samples.add(nd);
		}

		DataUtils.writeCSV(fn, samples, new String[] { "x", "y", "value", "class" });
	}

	public static void main(String[] args) {

		// buildDiamond(400, "data/geosomeval/diamond.csv", false);
		// buildSquareville( "data/geosomeval/squareville.csv");
		// buildSquarevilleX( "data/geosomeval/squarevilleX.csv");

		// buildTTest("data/geosomeval/ttest.csv");
		// buildConcentricCircles(400, 3, "data/geosomeval/circles.csv" );
		// buildOutlier("data/geosomeval/outlier.csv");
		// build4Corners("data/geosomeval/4corners.csv");
		// buildGaussianMixture( 400, "data/geosomeval/gm.csv");

		// buildNoisyCorners("data/geosomeval/noisycorners.csv");
		// buildVarCorners("data/geosomeval/varcorners.csv");

		// buildTwoPlanes("data/twoplanes.csv");
		// buildGaussianMixture(1000, "output/gm.csv");

		buildSyntheticSurface(81, "data/syn_surface.csv");

		// buildConcentricCircles(2000, 7, "output/circles.shp");

		/*
		 * buildRandomSamples(100, 6, "data/regionalization/100rand.shp"); List<double[]> samples = DataUtils.readSamplesFromShapeFile( new File("data/regionalization/100rand.shp"), new int[]{}, false); List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/regionalization/100rand.shp")); Map<double[],Set<double[]>> cm = RegionUtils.deriveQueenContiguitiyMap(samples, geoms ); RegionUtils.writeContiguityMap( cm, samples, "data/regionalization/100rand.ctg" );
		 */

		// buildJTest(1000, "output/test1.shp");
		// buildRandomRegions(8, 1000, 0, 4, "output/test2.shp");
		// buildNSquares(new double[]{0.49, 0.02, 0.49 }, 1000, 1, true, "output/test.shp");
		// buildNSquares(new double[]{1.0/3, 1.0/3, 1.0/3 }, new double[]{0.1,0.1,0.1}, 1000, 1, true, "output/bacao.shp");

		// buildNSquares(new double[]{ 0.1, 0.4, 0.5 }, 1000, 1, true, "output/bacao2.shp");

		// buildNSquares(new double[]{0.3,0.3,0.3,0.1,0.3}, 1000, 1, false, "output/5squares_neq.shp");

		// List<double[]> samples = readSamplesFromFcps("data/fcps/Tetra.lrn", "data/fcps/Tetra.cls");
		// DataUtils.writeCSV("output/tetra.csv", samples, new String[]{"x","y","z","c"});
		// buildRegioSamplesFromFcps("data/fcps/EngyTime.lrn",
		// "data/fcps/EngyTime.cls", "output/engytime.shp");
		// buildRegioSamplesFromFcps("data/fcps/Target.lrn",
		// "data/fcps/Target.cls", "output/target.shp");
		// buildRegioSamplesFromFcps("data/fcps/TwoDiamonds.lrn",
		// "data/fcps/TwoDiamonds.cls", "output/twodiamonds.shp");
		// buildRegioSamplesFromFcps("data/fcps/WingNut.lrn",
		// "data/fcps/WingNut.cls", "output/wingnut.shp");
		// buildRandomDiffDensRegions(25, "output/diffdensreg.shp");
		// buildNDiffDensSquares( 5, 2000, 0.05, 0.5,
		// "output/ndiffdenssquares.shp"); // rocks
		// buildNDiffDensSquares( 3, 2000, 0.025, 0.5, 2,1,"output/diffdens.shp"); // totally rocks!
		// buildNDiffDensSquares( 3, 2000, 0.025, 0.1, 2, 1,"data/cng/diffdens.shp"); // rocks?

		// buildNDiffDensSquares( 3, 2000, 0.5, 0.0, 1, 1,"output/test1.shp");
		// buildRandomRegions(25, 1000, 0.2, "output/test2a.shp");

		/*
		 * buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_01_001.csv"); buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_015_001.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_02_001.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_025_001.csv");
		 * 
		 * buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_01_0015.csv"); buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_015_0015.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_02_0015.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_025_0015.csv");
		 * 
		 * buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_01_002.csv"); buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_015_002.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_02_002.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_025_002.csv");
		 * 
		 * buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_01_0025.csv"); buildSPSDataSet( 100000, 0.1, 0.01, "data/sps/minkocube_015_0025.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_02_0025.csv"); buildSPSDataSet( 100000, 0.2, 0.01, "data/sps/minkocube_025_0025.csv");
		 */

		// buildAustriaSet();

		// buildSquareTest("data/adapt/4squares.shp");
		// buildSquareTest2("data/adapt/2squares.shp");
		// buildArtifialSpatialDataSet(1000, "data/adapt/plane.shp");

		// buildNSquares(new double[]{ 3f/4, 1f/8, 1f/8 }, 1000, 1f/8, true, "data/adapt/skewedBacao.shp");
		// buildNSquares(new double[]{ 0.3333, 0.3333, 0.3333 }, new double[]{0,0,0}, 1000, 1, true, "data/adapt/testBacao.shp");
		// buildTTest("output/tData.shp");
		// buildNSquares(new double[]{ 1f/3, 1f/3, 1f/3 }, new double[]{0,0,0}, 1000, 1f, true, "output/bacao.shp");
		// buildDiamond(1000, "output/diamond.shp", true );

		// List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/adapt/plane.shp"), new int[] {}, true);
		// DataUtils.writeCSV("output/plane.csv",samples,new String[]{"x","y","v"});

		// SampleBuilder.buildNDiffDensSquares(3,1000,0.5,0,1,1,"output/bacao.shp");

		// buildSquareTest3("data/adapt/4squares.shp");
		// buildTTest("data/adapt/tTest.shp");

		// buildRandomRegions(25, 1000, 0.2, "data/cng/test2a.shp");
		/*
		 * buildRandomRegions(25, 1000, 0.0, "data/cng/test2a_nonoise.shp");
		 * 
		 * final List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/cng/test2a_nonoise.shp"), new int[] {}, true); final List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/cng/test2a_nonoise.shp")); final Map<Integer,Set<double[]>> classes = new HashMap<Integer,Set<double[]>>(); for( double[] d : samples ) { int c = (int)d[3]; if( !classes.containsKey(c) ) classes.put(c, new HashSet<double[]>() ); classes.get(c).add(d); } try { Drawer.geoDrawCluster(classes.values(), samples, geoms, new FileOutputStream("output/cluster.png"), true); } catch (FileNotFoundException e) { // TODO Auto-generated catch block e.printStackTrace(); }
		 */

		/*
		 * buildConcentricCircles(2000, 25, "data/cng_var_test/circle1.shp" ); buildConcentricCircles(2000, 15, "data/cng_var_test/circle2.shp" );
		 * 
		 * buildRandomDiffDensRegions(25, "data/cng_var_test/diffdens1.shp"); buildRandomDiffDensRegions(15, "data/cng_var_test/diffdens2.shp"); buildRandomDiffDensRegions(25, "data/cng_var_test/diffdens3.shp");
		 * 
		 * buildRandomRegions(25, 2000, 0.1, "data/cng_var_test/regions1.shp"); buildRandomRegions(25, 2000, 0.25, "data/cng_var_test/regions2.shp");
		 */

		// buildNDiffDensSquares(15, 1000, 0.025, 0.5, 2, 1, "data/cng_var_test/diffsqr1.shp");
		// buildNDiffDensSquares(15, 1000, 0.025, 0.1, 2, 1, "data/cng_var_test/diffsqr2.shp");

		// buildRandomRegions(100, 5000, 0.1, 6, "data/cng_var_test/100regions.shp");

		/*
		 * try { List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/Chainlink.lrn", "data/fcps/Chainlink.cls"); DataUtil.writeCSV(new FileOutputStream("output/chainlink.csv"), samples); } catch (FileNotFoundException e) { e.printStackTrace(); }
		 * 
		 * try { List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/Hepta.lrn", "data/fcps/Hepta.cls"); DataUtil.writeCSV(new FileOutputStream("output/hepta.csv"), samples); } catch (FileNotFoundException e) { e.printStackTrace(); }
		 * 
		 * try { List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/GolfBall.lrn", "data/fcps/GolfBall.cls"); DataUtil.writeCSV(new FileOutputStream("output/golfball.csv"), samples); } catch (FileNotFoundException e) { e.printStackTrace(); }
		 * 
		 * try { List<double[]> samples = SampleBuilder.readSamplesFromFcps("data/fcps/Target.lrn", "data/fcps/Target.cls"); DataUtil.writeCSV(new FileOutputStream("output/target.csv"), samples); } catch (FileNotFoundException e) { e.printStackTrace(); }
		 */

		/*
		 * List<double[]> samples = DataUtil.readSamplesFromShapeFile( new File("data/lisbon/lisbon.shp"), new int[]{}, false); List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/lisbon/lisbon.shp")); Map<double[],Set<double[]>> cm = RegionUtils.deriveQueenContiguitiyMap(samples, geoms ); RegionUtils.writeContiguityMap( cm, samples, "data/lisbon/lisbon_queen.ctg" );
		 */

		// buildNDiffDensSquares( 3, 1000, 0.01, 0.2, 1.0, 1.0,"output/test.shp");

		// buildGaussian(1000, 4, "output/gaussPoints.shp");
		// List<double[]> samples = DataUtils.readSamplesFromShapeFile( new File("output/gaussPoints.shp"), new int[]{}, true);
		// DataUtils.writeCSV( "output/gaussPoints.csv", samples, new String[]{"x","y","v"});

	}
}
