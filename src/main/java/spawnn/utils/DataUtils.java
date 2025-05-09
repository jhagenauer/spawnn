package spawnn.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.geotools.data.DataStore;
import org.geotools.data.DefaultTransaction;
import org.geotools.data.FeatureSource;
import org.geotools.data.Transaction;
import org.geotools.data.collection.ListFeatureCollection;
import org.geotools.data.shapefile.ShapefileDataStore;
import org.geotools.data.shapefile.ShapefileDataStoreFactory;
import org.geotools.data.simple.SimpleFeatureSource;
import org.geotools.data.simple.SimpleFeatureStore;
import org.geotools.feature.FeatureIterator;
import org.geotools.feature.simple.SimpleFeatureBuilder;
import org.geotools.feature.simple.SimpleFeatureTypeBuilder;
import org.jblas.DoubleMatrix;
import org.opengis.feature.simple.SimpleFeature;
import org.opengis.feature.simple.SimpleFeatureType;
import org.opengis.feature.type.AttributeDescriptor;
import org.opengis.referencing.crs.CoordinateReferenceSystem;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryCollection;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.MultiPoint;
import com.vividsolutions.jts.geom.MultiPolygon;
import com.vividsolutions.jts.geom.Point;
import com.vividsolutions.jts.geom.Polygon;
import com.vividsolutions.jts.io.ParseException;
import com.vividsolutions.jts.io.WKTReader;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.utils.DataFrame.binding;

public class DataUtils {

	private static Logger log = LogManager.getLogger(DataUtils.class);

	public static List<double[]> mappingToCluster(Map<double[], Set<double[]>> m) {
		List<double[]> r = new ArrayList<double[]>();

		List<double[]> k = new ArrayList<double[]>(m.keySet());
		for (int i = 0; i < k.size(); i++) {
			for (double[] d : m.get(k.get(i))) {
				double[] nd = Arrays.copyOf(d, d.length + 1);
				nd[nd.length - 1] = i;
				r.add(nd);
			}
		}
		return r;
	}

	public static List<double[]> getSammonsProjection(List<double[]> samples, Dist<double[]> a, Dist<double[]> b, int dim) {
		List<double[]> projected = new ArrayList<double[]>();

		// init
		for (double[] d : samples)
			projected.add(Arrays.copyOf(d, dim));

		for (double lambda = 1; lambda >= 0.01; lambda *= 0.95) {

			for (int i = 0; i < samples.size(); i++) {
				for (int j = 0; j < samples.size(); j++) {
					if (i == j)
						continue;

					double dStar = a.dist(samples.get(i), samples.get(j));
					double d = b.dist(projected.get(i), projected.get(j));

					if (dStar == 0)
						dStar = 1e-10;

					double delta = lambda * (dStar - d) / dStar;
					for (int k = 0; k < dim; k++) {
						double correction = delta * (projected.get(i)[k] - projected.get(j)[k]);
						projected.get(i)[k] += correction;
						projected.get(j)[k] -= correction;
					}
					System.out.println(delta + ":" + dStar + ":" + d);

				}
			}
		}
		return projected;
	}

	public static int countCompactClusters(Collection<List<double[]>> cluster, List<Geometry> geoms,
			List<double[]> samples) {
		int sum = 0;
		GeometryFactory gf = new GeometryFactory();

		for (List<double[]> l : cluster) {
			if (l.size() == 0)
				continue;
			Geometry[] gs = new Geometry[l.size()];
			for (int i = 0; i < l.size(); i++)
				gs[i] = geoms.get(samples.indexOf(l.get(i)));
			GeometryCollection gc = gf.createGeometryCollection(gs);
			if (gc.union().getNumGeometries() == 1)
				sum++;
		}
		return sum;
	}

	public static long binCoeff(int n, int k) {
		long[] b = new long[n + 1];
		b[0] = 1;
		for (int i = 1; i <= n; i++) {
			b[i] = 1;
			for (int j = i - 1; j > 0; j--)
				b[j] += b[j - 1];
		}
		if (k > b.length - 1)
			return 0;
		else
			return b[k];
	}

	// does not work as exspected
	public static double getAdjustedRandIndex(List<List<double[]>> u, List<List<double[]>> v) {

		int[][] contTable = new int[u.size()][v.size()];
		for (int i = 0; i < u.size(); i++) { // zeile
			for (int j = 0; j < v.size(); j++) { // spalte
				Set<double[]> s = new HashSet<double[]>();
				s.addAll(u.get(i));
				s.addAll(v.get(j));
				contTable[i][j] = s.size();
			}
		}

		int[] aSums = new int[contTable.length];
		for (int i = 0; i < contTable.length; i++) { // zeile
			int a = 0;
			for (int j = 0; j < contTable[i].length; j++)
				// spalte
				a += contTable[i][j];
			aSums[i] = a;
		}

		int[] bSums = new int[contTable[0].length];
		for (int j = 0; j < contTable[0].length; j++) { // spalte
			int b = 0;
			for (int i = 0; i < contTable.length; i++)
				// zeile
				b += contTable[i][j];
			bSums[j] = b;
		}

		int n = 0;
		for (int i : aSums)
			n += i;

		double index = 0;
		for (int i = 0; i < contTable.length; i++)
			for (int j = 0; j < contTable[i].length; j++)
				index += binCoeff(contTable[i][j], 2);

		double coSumA = 0;
		for (int i = 0; i < aSums.length; i++)
			coSumA += binCoeff(aSums[i], 2);

		double coSumB = 0;
		for (int i = 0; i < bSums.length; i++)
			coSumB += binCoeff(bSums[i], 2);

		double exspectedIndex = (coSumA * coSumB) / binCoeff(n, 2);
		double maxIndex = 0.5 * (coSumA + coSumB);

		return (index - exspectedIndex) / (maxIndex - exspectedIndex);
	}

	public static double getDaviesBouldinIndex(Collection<Set<double[]>> clusters, Dist<double[]> dist) {
		double sum = 0;

		for (Set<double[]> c1 : clusters) {
			if (c1.isEmpty())
				continue;

			double[] c1Center = getMean(c1);

			double max = Double.MIN_VALUE;
			for (Set<double[]> c2 : clusters) {
				if (c2.isEmpty())
					continue;

				double[] c2Center = getMean(c2);

				if (c1.equals(c2))
					continue;

				double si = 0;
				for (double[] d : c1)
					si += dist.dist(d, c1Center);
				si /= c1.size();

				double sj = 0;
				for (double[] d : c2)
					sj += dist.dist(d, c2Center);
				sj /= c2.size();

				double r = (si + sj) / dist.dist(c1Center, c2Center);

				if (r > max)
					max = r;
			}
			sum += max;
		}
		return sum / clusters.size();
	}

	public static double[] getMean(Collection<double[]> c) {
		int l = c.iterator().next().length;
		double[] r = new double[l];
		for (double[] d : c)
			for (int i = 0; i < l; i++)
				r[i] += d[i];
		for (int i = 0; i < l; i++)
			r[i] /= c.size();
		return r;
	}

	public static double getSumOfSquares(Map<double[], Set<double[]>> clusters, Dist<double[]> dist) {
		double totalSum = 0;
		for (Entry<double[], Set<double[]>> e : clusters.entrySet())
			totalSum += getSumOfSquares(e.getKey(), e.getValue(), dist);
		return totalSum;
	}

	public static <T> double getSumOfSquares(T center, Set<T> s, Dist<T> dist) {
		double sum = 0;
		for (T d : s)
			sum += Math.pow(dist.dist(d, center), 2);
		return sum;
	}

	public static <T> double getMeanQuantizationError(Map<T, Set<T>> clusters, Dist<T> dist) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (Entry<T, Set<T>> e : clusters.entrySet())
			for (T d : e.getValue())
				ds.addValue(dist.dist(e.getKey(), d));
		return ds.getMean();
	}

	public static <T> double getQuantizationError(T centroid, Collection<T> data, Dist<T> dist) {
		double sum = 0;
		for (T d : data)
			sum += dist.dist(centroid, d);
		return sum / data.size();
	}

	public static void writeShape(List<double[]> samples, List<Geometry> geoms, String fn) {
		writeShape(samples, geoms, null, null, fn);
	}

	public static void writeShape(List<double[]> samples, List<Geometry> geoms, String[] names, String fn) {
		writeShape(samples, geoms, names, null, fn);
	}

	public static void writeShape(List<double[]> samples, List<Geometry> geoms, String[] names,
			CoordinateReferenceSystem crs, String fn) {
		if (names != null && samples.get(0).length != names.length)
			throw new RuntimeException(
					"column-length does not match names-length: " + samples.get(0).length + "!=" + names.length);

		try {
			SimpleFeatureTypeBuilder typeBuilder = new SimpleFeatureTypeBuilder();
			typeBuilder.setName("samples");
			for (int i = 0; i < samples.get(0).length; i++) {
				if (names != null) {
					String n = names[i];
					if (n.length() > 8) {
						n = n.substring(0, 8);
						// log.debug("Trunkating " + names[i] + " to " + n);
					}
					typeBuilder.add(n, Double.class);
				} else {
					typeBuilder.add("data" + i, Double.class);
				}
			}

			if (crs != null)
				typeBuilder.setCRS(crs);
			else
				log.warn("CRS not set!");

			Geometry g = geoms.get(0);
			if (g instanceof Polygon)
				typeBuilder.add("the_geom", Polygon.class);
			else if (g instanceof MultiPolygon)
				typeBuilder.add("the_geom", MultiPolygon.class);
			else if (g instanceof Point)
				typeBuilder.add("the_geom", Point.class);
			else if (g instanceof MultiPoint)
				typeBuilder.add("the_geom", MultiPoint.class);
			else
				throw new RuntimeException("Unkown geometry type!");

			SimpleFeatureType type = typeBuilder.buildFeatureType();
			SimpleFeatureBuilder featureBuilder = new SimpleFeatureBuilder(type);

			ListFeatureCollection fc = new ListFeatureCollection(type);
			for (double[] d : samples) {
				for (int i = 0; i < d.length; i++)
					featureBuilder.add(d[i]);
				featureBuilder.add(geoms.get(samples.indexOf(d)));
				SimpleFeature sf = featureBuilder.buildFeature(null);
				fc.add(sf);
			}

			ShapefileDataStoreFactory dataStoreFactory = new ShapefileDataStoreFactory();

			Map<String, Serializable> params = new HashMap<>();
			params.put("url", new File(fn).toURI().toURL());
			params.put("create spatial index", Boolean.TRUE);

			ShapefileDataStore newDataStore = (ShapefileDataStore) dataStoreFactory.createNewDataStore(params);
			newDataStore.createSchema(fc.getSchema());

			String typeName = newDataStore.getTypeNames()[0];
			SimpleFeatureSource featureSource = newDataStore.getFeatureSource(typeName);
			SimpleFeatureStore featureStore = (SimpleFeatureStore) featureSource;

			Transaction transaction = new DefaultTransaction("create");
			try {
				featureStore.addFeatures(fc);
				transaction.commit();
			} catch (Exception e) {
				e.printStackTrace();
				transaction.rollback();
			} finally {
				transaction.close();
			}

		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Deprecated
	public static List<Geometry> readGeometriesFromShapeFile(File file) {
		return readShapedata(file, new int[] {}, false).geoms;
	}

	@Deprecated
	public static List<double[]> readSamplesFromShapeFile(File file, int[] ign, boolean printAttrs) {
		return readShapedata(file, ign, printAttrs).samples;
	}

	@Deprecated
	public static SpatialDataFrame readShapedata(File file, int[] ign, boolean printAttrs) {
		return readSpatialDataFrameFromShapefile(file, printAttrs);
	}

	public static List<Geometry> readCSVGeometries(InputStream is, int col) {
		List<Geometry> geoms = new ArrayList<Geometry>();

		BufferedReader reader = new BufferedReader(new InputStreamReader(is));

		WKTReader wr = new WKTReader();

		try {
			// header handling
			String header = null;
			while ((header = reader.readLine()) != null)
				// skip comments
				if (!header.startsWith("#")) // header found
					break;

			String line = null;
			while ((line = reader.readLine()) != null) {

				if (line.startsWith("#"))
					continue;

				String g = line.split(",")[col];
				geoms.add(wr.read(g));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}

		return geoms;
	}

	@Deprecated
	public static List<double[]> readCSV(String file, int[] ign) {
		try {
			return readCSV(new FileInputStream(file), ign);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Deprecated
	public static List<double[]> readCSV(String file) {
		return readCSV(file, new int[] {});
	}

	@Deprecated
	public static List<double[]> readCSV(InputStream is) {
		return readCSV(is, new int[] {});
	}

	@Deprecated
	public static List<double[]> readCSV(InputStream is, int[] ign) {
		SpatialDataFrame sd = new SpatialDataFrame(); // actually, its not spatial data
		List<double[]> r = new ArrayList<double[]>();

		Set<Integer> ignore = new HashSet<Integer>();
		for (int i : ign)
			ignore.add(i);

		BufferedReader reader = new BufferedReader(new InputStreamReader(is));

		try {
			// header handling
			String header = null;
			while ((header = reader.readLine()) != null)
				// skip comments
				if (!header.startsWith("#"))
					break;

			List<String> names = new ArrayList<String>();
			String[] h = header.split(",");

			for (int i = 0; i < h.length; i++) {
				if (!ignore.contains(i))
					names.add(h[i]);
			}
			sd.names = names;

			String line = null;
			while ((line = reader.readLine()) != null) {

				if (line.startsWith("#"))
					continue;

				String[] data = line.split(",");
				double[] d = new double[data.length - ign.length];
				int modIdx = 0;
				for (int i = 0; i < data.length; i++)
					if (ignore.contains(i))
						modIdx++;
					else
						d[i - modIdx] = Double.parseDouble(data[i]);
				r.add(d);

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		sd.samples = r;

		return sd.samples;
	}
	
	public static void writeCSV_Double(OutputStream os, List<Double> samples) {
		List<double[]> l = new ArrayList<>();
		for( Double d : samples )
			l.add( new double[] {d});
		writeCSV(os, l, null, ',');
	}

	public static void writeCSV(OutputStream os, List<double[]> samples) {
		writeCSV(os, samples, null, ',');
	}

	public static void writeCSV(String fn, Collection<double[]> samples, String[] names) {
		writeCSV(fn, samples, names, ',');
	}

	public static void writeCSV(String fn, Collection<double[]> samples, String[] names, char sep) {
		try {
			writeCSV(new FileOutputStream(fn), samples, names, sep);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public static void writeMatrix(String fn, DoubleMatrix m ) {
		try {
			String h = "X1";
			for( int i = 1; i < m.columns; i++ )
				h+=",X"+i;
			h+="\n";
			
			Files.write(Paths.get(fn), h.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
			for( int i = 0; i < m.rows; i++ ) {
				String s = Arrays.toString(m.getRow(i).data).replaceAll("\\[", "").replaceAll("\\]", "")+"\n";
				Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);								
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void writeCSV(OutputStream os, Collection<double[]> samples, String[] names, char sep) {
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(os));

			if (names != null) {
				String header = "";
				for (int i = 0; i < names.length; i++)
					header += names[i] + sep;
				header = header.substring(0, header.length() - 1);
				writer.write(header + "\n");
			} else {
				String header = "";
				for (int i = 0; i < samples.iterator().next().length; i++)
					header += "att_" + i + sep;
				header = header.substring(0, header.length() - 1);
				writer.write(header + "\n");
			}

			// not really nice
			Map<Integer, Boolean> typeMap = new HashMap<Integer, Boolean>();
			for (int i = 0; i < samples.iterator().next().length; i++) {
				boolean integer = true;
				for (double[] d : samples)
					if ((d[i] != Math.floor(d[i])) || Double.isInfinite(d[i]))
						integer = false;
				typeMap.put(i, integer);
			}

			for (double[] d : samples) {
				String s = "";
				for (int i = 0; i < d.length; i++)
					if (typeMap.get(i))
						s += (int) d[i] + "" + sep;
					else
						s += d[i] + "" + sep;
				s = s.substring(0, s.length() - 1);
				writer.write(s + "\n");
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static DataFrame readDataFrameFromCSV(File file, int[] ign, boolean verbose) {
		try {
			InputStream is = new FileInputStream(file);
			DataFrame ds = readDataFrameFromInputStream(is, ign, verbose, ',');
			is.close();
			return ds;
		} catch ( IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static DataFrame readDataFrameFromInputStream(InputStream is, int[] ign, boolean verbose, char sep) {
		char quote = '"';

		Set<Integer> ignored = new HashSet<Integer>();
		for (int i : ign)
			ignored.add(i);

		DataFrame sd = new DataFrame();
		
		List<double[]> r = new ArrayList<double[]>();
			
		BufferedReader reader = null;
		String ssep = sep + "";
		try {
			reader = new BufferedReader(new InputStreamReader(is));
				
			// header handling
			String header = null;
			while ((header = reader.readLine()) != null)
				// skip comments
				if (!header.startsWith("#"))
					break;

			// clear line from commas within strings
			{
				boolean start = false;
				char[] charStr = header.toCharArray();
				for (int i = 0; i < charStr.length; i++) {
					if (charStr[i] == quote)
						start = !start;
					if (start == true && charStr[i] == sep)
						charStr[i] = ';'; // replace
				}
				header = String.valueOf(charStr);
			}
			String[] h = header.split(ssep);
						
			int j = 0;
			String line = null;
			while ((line = reader.readLine()) != null) {

				if (line.startsWith("#"))
					continue;

				// clear line from commas within strings
				boolean start = false;
				char[] charStr = line.toCharArray();
				for (int i = 0; i < charStr.length; i++) {
					if (charStr[i] == quote)
						start = !start;
					if (start == true && charStr[i] == sep)
						charStr[i] = ';'; // replace
				}

				line = String.valueOf(charStr);
				String[] data = line.split(sep + "");
																				
				double[] d = new double[data.length];
				for (int i = 0; i < data.length; i++) {
					if (ignored.contains(i))
						continue;
					else if( data[i].isEmpty() )
						d[i] = Double.NaN;
					else {
						try {
							d[i] = Double.parseDouble(data[i]);
						} catch (NumberFormatException e) {
							log.warn("Cannot parse value " + data[i] + " in column " + i + ", row "+j+", ignoring column "+h[i]+"..."+e.getMessage());
							ignored.add(i);
						}
					}
				}			
				r.add(d);
				j++;
			}
			
			// build final samples
			List<String> names = new ArrayList<>();
			List<Integer> notIgnored = new ArrayList<>();
			for( int i = 0; i < h.length; i++ )
				if( !ignored.contains(i) ) { 
					notIgnored.add(i);
					names.add(h[i]);
				}
			int[] notIgnoredArray = new int[notIgnored.size()];
			for( int i = 0; i < notIgnored.size(); i++ )
				notIgnoredArray[i] = notIgnored.get(i);
			
			// do we need to strip
			if( !ignored.isEmpty() ) {
				List<double[]> nr = new ArrayList<>();
				while( !r.isEmpty() ) {
					double[] d = r.remove(0);
					nr.add( DataUtils.strip(d, notIgnoredArray ));
				}			
				sd.samples = nr;
			} else 
				sd.samples = r;
			sd.names  = names;
				
			// TODO for now its all double
			sd.bindings = new ArrayList<binding>();
			for (int i = 0; i < sd.names.size(); i++) {
				if( verbose ) log.debug(i+","+sd.names.get(i) );
				sd.bindings.add(binding.Double);
			}			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null)
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}		
		return sd;
	}
	
	public static DataFrame2 readDataFrame2FromInputStream(InputStream is, char sep ) {
		char quote = '"';
		DataFrame2 sd = new DataFrame2();
					
		BufferedReader reader = null;
		String ssep = sep + "";
		try {
			reader = new BufferedReader(new InputStreamReader(is));
				
			// header handling
			String header = null;
			while ((header = reader.readLine()) != null)
				// skip comments
				if (!header.startsWith("#"))
					break;

			// clear line from commas within strings
			{
				boolean start = false;
				char[] charStr = header.toCharArray();
				for (int i = 0; i < charStr.length; i++) {
					if (charStr[i] == quote)
						start = !start;
					if (start == true && charStr[i] == sep)
						charStr[i] = ';'; // replace
				}
				header = String.valueOf(charStr);
			}
			String[] h = header.split(ssep);
						
			String line = null;
			List<String[]> data = new ArrayList<>();
			while ((line = reader.readLine()) != null) {

				if (line.startsWith("#"))
					continue;

				// clear line from commas within strings
				boolean start = false;
				char[] charStr = line.toCharArray();
				for (int i = 0; i < charStr.length; i++) {
					if (charStr[i] == quote)
						start = !start;
					if (start == true && charStr[i] == sep)
						charStr[i] = ';'; // replace
				}
				line = String.valueOf(charStr);
				String[] d = line.split(sep + "");
				if( d.length != h.length )
					throw new RuntimeException("Columns header "+h.length+" != columns data "+d.length);
				data.add( d );					
			}
						
			List<double[]> cols_num = new ArrayList<>();
			List<String[]> cols_str = new ArrayList<>();
			List<String> names_num = new ArrayList<>();
			List<String> names_str = new ArrayList<>();		
			
			for( int i = 0; i < h.length; i++ ) { // each col
				String[] cs = new String[data.size()];
				double[] cd = new double[data.size()];
				
				boolean use_str = false;
				for( int j = 0; j < data.size(); j++ ) {
					String d = data.get(j)[i];
					if( d.isEmpty() )
						cd[j] = Double.NaN;
					else {
						try {
							cd[j] = Double.parseDouble(d);
						} catch (NumberFormatException e) {
							cs[j] = d;
							use_str = true;							
						}
					}
				}
				
				if( use_str ) {
					cols_str.add(cs);
					names_str.add(h[i]);
				} else {
					cols_num.add(cd);
					names_num.add(h[i]);
				}
			}
			
			// names
			sd.names_num = new String[names_num.size()];
			for( int i = 0; i < sd.names_num.length; i++ )
				sd.names_num[i] = names_num.get(i);
			
			sd.names_str = new String[names_str.size()];
			for( int i = 0; i < sd.names_str.length; i++ )
				sd.names_str[i] = names_str.get(i);

			// columns to rows
			sd.samples_num = new double[cols_num.get(0).length][cols_num.size()];
			sd.samples_str = new String[cols_num.get(0).length][cols_str.size()];
			for( int j = 0; j < cols_num.get(0).length; j++ ) { // for each sample
				
				double[] d = new double[cols_num.size()];
				for( int i = 0; i < d.length; i++ )
					sd.samples_num[j][i] = cols_num.get(i)[j];
				
				String[] s = new String[cols_str.size()];
				for( int i = 0; i < s.length; i++ )
					sd.samples_str[j][i] = cols_str.get(i)[j];
			}			
				
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null)
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}		
		return sd;
	}
	
	public static DoubleMatrix readDoubleMatrixFromInputStream(InputStream is,char sep) {		
		List<double[]> r = new ArrayList<double[]>();			
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(is));			
			String line = null;
			while ((line = reader.readLine()) != null) {
				
				String[] data = line.split(sep + "");
				double[] d = new double[data.length];
				for (int i = 0; i < data.length; i++) {
					try {
						d[i] = Double.parseDouble(data[i]);
					} catch (NumberFormatException e) {
						e.printStackTrace();
						System.exit(1);
					}
				}			
				r.add(d);
			}		
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null)
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}
		
		DoubleMatrix d = new DoubleMatrix(r.size(),r.get(0).length);
		for( int i = 0; i < r.size(); i++ )
			d.putRow(i, new DoubleMatrix(r.get(i)));
		return d;
	}

	// FIXME verbose has no effect
	public static SpatialDataFrame readSpatialDataFrameFromCSV(File file, int[] ga, int[] ign, boolean verbose) {
		GeometryFactory gf = new GeometryFactory();

		SpatialDataFrame sd = new SpatialDataFrame();
		sd.samples = new ArrayList<double[]>();
		sd.geoms = new ArrayList<Geometry>();
		sd.bindings = new ArrayList<SpatialDataFrame.binding>();

		Set<Integer> ignore = new HashSet<Integer>();
		for (int i : ign)
			ignore.add(i);

		BufferedReader reader = null;

		try {
			reader = new BufferedReader(new FileReader(file));
			// header handling
			String header = null;
			while ((header = reader.readLine()) != null)
				// skip comments
				if (!header.startsWith("#"))
					break;

			List<String> names = new ArrayList<String>();
			String[] h = header.split(",");

			for (int i = 0; i < h.length; i++) {
				if (!ignore.contains(i)) {
					log.debug(i + "," + h[i] + "," + names.size());
					names.add(h[i]);
				}
			}
			sd.names = names;

			String line = null;
			while ((line = reader.readLine()) != null) {

				if (line.startsWith("#"))
					continue;

				String[] data = line.split(",");
				double[] d = new double[data.length - ign.length];
				int modIdx = 0;
				for (int i = 0; i < data.length; i++)
					if (ignore.contains(i))
						modIdx++;
					else
						d[i - modIdx] = Double.parseDouble(data[i]);

				sd.samples.add(d);

				Coordinate c;
				if (ga.length == 2) {
					c = new Coordinate(Double.parseDouble(data[ga[0]]), Double.parseDouble(data[ga[1]]));
				} else if (ga.length == 3) {
					c = new Coordinate(Double.parseDouble(data[ga[0]]), Double.parseDouble(data[ga[1]]),
							Double.parseDouble(data[ga[2]]));
				} else {
					throw new RuntimeException("Cannot create point of length " + ga.length);
				}
				sd.geoms.add(gf.createPoint(c));
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null)
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}

		// TODO for now its all double
		sd.bindings = new ArrayList<binding>();
		for (int i = 0; i < sd.names.size(); i++)
			sd.bindings.add(binding.Double);

		return sd;
	}

	public static SpatialDataFrame readSpatialDataFrameFromShapefile(File file, boolean debug) {
		return readSpatialDataFrameFromShapefile(file, new int[] {}, debug);
	}

	public static SpatialDataFrame readSpatialDataFrameFromShapefile(File file, int[] toDouble, boolean debug) {

		SpatialDataFrame sd = new SpatialDataFrame();
		sd.samples = new ArrayList<double[]>();
		sd.geoms = new ArrayList<Geometry>();
		sd.names = new ArrayList<String>();
		sd.bindings = new ArrayList<SpatialDataFrame.binding>();

		DataStore dataStore = null;
		try {
			dataStore = new ShapefileDataStore((file).toURI().toURL());
			FeatureSource<SimpleFeatureType, SimpleFeature> featureSource = dataStore
					.getFeatureSource(dataStore.getTypeNames()[0]);
			sd.crs = featureSource.getSchema().getCoordinateReferenceSystem();

			Set<Integer> td = new HashSet<>();
			for (int i : toDouble)
				td.add(i);
			Set<Integer> ignore = new HashSet<Integer>();

			List<AttributeDescriptor> adl = featureSource.getFeatures().getSchema().getAttributeDescriptors(); // all
			for (int i = 0; i < adl.size(); i++) {
				AttributeDescriptor ad = adl.get(i);
				String bin = ad.getType().getBinding().getName();

				if (ignore.contains(i))
					continue;

				if (bin.equals("java.lang.Integer")) {
					sd.names.add(ad.getLocalName());
					sd.bindings.add(SpatialDataFrame.binding.Integer);
				} else if (bin.equals("java.lang.Double") || td.contains(i)) {
					sd.names.add(ad.getLocalName());
					sd.bindings.add(SpatialDataFrame.binding.Double);
				} else if (bin.equals("java.lang.Long")) {
					sd.names.add(ad.getLocalName());
					sd.bindings.add(SpatialDataFrame.binding.Long);
				} else {
					ignore.add(i);
					if (debug)
						log.debug("Ignoring " + ad.getLocalName() + ", because " + bin);
				}
			}

			if (debug) {
				int idx = 0;
				for (int i = 0; i < adl.size(); i++) {
					if (ignore.contains(i))
						log.debug(i + ":" + adl.get(i).getLocalName() + ", IGN");
					else
						log.debug(i + ":" + adl.get(i).getLocalName() + "," + (idx++));
				}
			}

			FeatureIterator<SimpleFeature> it = featureSource.getFeatures().features();
			try {
				while (it.hasNext()) {
					SimpleFeature feature = it.next();
					double[] d = new double[adl.size() - ignore.size()];

					int idx = 0;
					for (int i = 0; i < adl.size(); i++) {

						if (ignore.contains(i))
							continue;

						String name = adl.get(i).getLocalName();
						Object o = feature.getAttribute(name);
						if (td.contains(i) && o instanceof String)
							d[idx++] = Double.parseDouble((String) o);
						else if (o instanceof Double)
							d[idx++] = ((Double) o).doubleValue();
						else if (o instanceof Integer)
							d[idx++] = ((Integer) o).intValue();
						else if (o instanceof Long)
							d[idx++] = ((Long) o).longValue();
						else {
							log.error("Unknown attribute type: " + name + ", " + o + ", " + i);
							System.exit(1);
						}
					}

					sd.samples.add(d);
					sd.geoms.add((Geometry) feature.getDefaultGeometry());
				}
			} finally {
				if (it != null) {
					it.close();
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			dataStore.dispose();
		}
		return sd;
	}

	@Deprecated
	public static void normalize(List<double[]> samples) {
		int length = samples.get(0).length;
		for (int i = 0; i < length; i++)
			normalizeColumn(samples, i);
	}

	@Deprecated
	public static void normalizeColumn(List<double[]> samples, int idx) {
		SummaryStatistics ss = new SummaryStatistics();
		for (double[] d : samples)
			ss.addValue(d[idx]);

		for (double[] d : samples)
			d[idx] = (d[idx] - ss.getMin()) / (ss.getMax() - ss.getMin()); // 0...1
	}

	@Deprecated
	public static void normalizeColumns(List<double[]> samples, int[] idx) {
		for (int i : idx)
			normalizeColumn(samples, i);
	}

	public static void normalizeGeoColumns(List<double[]> samples, int[] idx) {
		double[] min = new double[idx.length], max = new double[idx.length];
		for (int i = 0; i < idx.length; i++) {
			min[i] = Double.MAX_VALUE;
			max[i] = Double.MIN_VALUE;
		}

		for (double[] d : samples) {
			for (int i = 0; i < idx.length; i++) {
				min[i] = Math.min(min[i], d[idx[i]]);
				max[i] = Math.max(max[i], d[idx[i]]);
			}
		}

		double l = Double.MIN_VALUE;
		for (int i = 0; i < idx.length; i++)
			l = Math.max(l, max[i] - min[i]);

		for (double[] d : samples)
			for (int i = 0; i < idx.length; i++)
				d[idx[i]] = (d[idx[i]] - min[i]) / l;
	}

	@Deprecated
	public static void zScore(List<double[]> samples) {
		int length = samples.get(0).length;
		for (int i = 0; i < length; i++)
			zScoreColumn(samples, i);
	}

	@Deprecated
	public static void zScoreColumn(List<double[]> samples, int idx) {
		SummaryStatistics ss = new SummaryStatistics();
		for (double[] d : samples)
			ss.addValue(d[idx]);

		for (double[] d : samples)
			d[idx] = (d[idx] - ss.getMean()) / ss.getStandardDeviation();
	}

	@Deprecated
	public static void zScoreColumns(List<double[]> samples) {
		int length = samples.get(0).length;
		for (int i = 0; i < length; i++)
			zScoreColumn(samples, i);
	}

	@Deprecated
	public static void zScoreColumns(List<double[]> samples, int[] idx) {
		for (int i : idx)
			zScoreColumn(samples, i);
	}

	public static void zScoreGeoColumns(List<double[]> samples, int[] idx, Dist<double[]> dist) {
		double[] mean = new double[samples.get(0).length];
		for (double[] s : samples)
			for (int i : idx)
				mean[i] += s[i] / samples.size();

		double stdDev = 0;
		for (double[] s : samples)
			stdDev += Math.pow(dist.dist(s, mean), 2);
		stdDev = Math.sqrt(stdDev / (samples.size() - 1));

		for (double[] s : samples)
			for (int i : idx)
				s[i] = (s[i] - mean[i]) / stdDev;
	}

	public static List<double[]> removeColumns(List<double[]> samples, int[] ign) {
		List<double[]> ns = new ArrayList<double[]>();

		Set<Integer> ignore = new HashSet<Integer>();
		for (int i : ign)
			ignore.add(i);

		for (double[] x : samples) {
			double[] n = new double[x.length - ign.length];

			int mod = 0;
			for (int i = 0; i < x.length; i++) {
				if (ignore.contains(i) || i >= n.length)
					mod++;
				else
					n[i - mod] = x[i];
			}
			ns.add(n);
		}
		return ns;
	}

	public static List<double[]> retainColumns(List<double[]> samples, int[] r) {
		int n = samples.get(0).length;

		// build list from array
		List<Integer> ret = new ArrayList<Integer>();
		for (int i : r)
			ret.add(i);

		// build all list
		List<Integer> idx = new ArrayList<Integer>();
		for (int i = 0; i < n; i++)
			idx.add(i);

		// remove wanted from all-list
		idx.removeAll(ret);

		// build array
		int[] a = new int[idx.size()];
		for (int i = 0; i < idx.size(); i++)
			a[i] = idx.get(i);

		// remove collums of all-,list
		return removeColumns(samples, a);
	}

	public static <T> Map<T, Map<T, Double>> readDistMatrixSquare(List<T> samples, File fn) {
		Map<T, Map<T, Double>> distMatrix = new HashMap<T, Map<T, Double>>();
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(fn));
			String line = null;
			int lineIdx = 0;
			while ((line = br.readLine()) != null) {
				Map<T, Double> m = new HashMap<T, Double>();

				String[] s = line.split(",");
				for (int i = 0; i < s.length; i++)
					if (!s[i].isEmpty()) {
						double d = Double.parseDouble(s[i]);
						m.put(samples.get(i), d);
					}
				distMatrix.put(samples.get(lineIdx++), m);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		return distMatrix;
	}

	public static double getSumOfSquares(Collection<double[]> s, Dist<double[]> dist) {
		double ssq = 0;
		if (s.isEmpty())
			return ssq;

		double[] mean = getMean(s);
		for (double[] d : s) {
			double di = dist.dist(mean, d);
			ssq += di * di;
		}

		return ssq;
	}

	public static double getWithinSumOfSquares(Collection<Set<double[]>> c, Dist<double[]> dist) {
		double ssq = 0;
		for (Set<double[]> s : c)
			ssq += getSumOfSquares(s, dist);
		return ssq;
	}

	// If one wants to perform PCA on a correlation matrix (instead of a
	// covariance matrix), then columns of X should not only be centered, but
	// standardized as well, i.e. divided by their standard deviations.
	public static List<double[]> reduceDimensionByPCA(List<double[]> samples, int nrComponents) {
		RealMatrix matrix = new Array2DRowRealMatrix(samples.size(), samples.get(0).length);
		for (int i = 0; i < samples.size(); i++)
			matrix.setRow(i, samples.get(i));

		SingularValueDecomposition svd = new SingularValueDecomposition(matrix);
		RealMatrix v = svd.getU().multiply(svd.getS());

		List<double[]> ns = new ArrayList<double[]>();
		for (int i = 0; i < v.getRowDimension(); i++) {
			double[] d = Arrays.copyOf(v.getRow(i), nrComponents);
			ns.add(d);
		}
		return ns;
	}

	public static double[] strip(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int j = 0; j < fa.length; j++)
			nd[j] = d[fa[j]];
		return nd;
	}

	public static <T> T concatenate(T a, T b) {
		if (!a.getClass().isArray() || !b.getClass().isArray()) {
			throw new IllegalArgumentException();
		}
	
		Class<?> resCompType;
		Class<?> aCompType = a.getClass().getComponentType();
		Class<?> bCompType = b.getClass().getComponentType();
	
		if (aCompType.isAssignableFrom(bCompType)) {
			resCompType = aCompType;
		} else if (bCompType.isAssignableFrom(aCompType)) {
			resCompType = bCompType;
		} else {
			throw new IllegalArgumentException();
		}
	
		int aLen = Array.getLength(a);
		int bLen = Array.getLength(b);
	
		@SuppressWarnings("unchecked")
		T result = (T) Array.newInstance(resCompType, aLen + bLen);
		System.arraycopy(a, 0, result, 0, aLen);
		System.arraycopy(b, 0, result, aLen, bLen);
	
		return result;
	}
	
	public static double[] concatenate( double[][] d ) {
		List<Double> l = new ArrayList<>();
		for( double[] a : d )
			for( double b : a )
				l.add(b);
		double[] r = new double[l.size()];
		for( int i = 0; i < l.size(); i++ )
			r[i] = l.get(i);			
		return r;
	}
	
	public static int[] concatenate( int[][] d ) {
		List<Integer> l = new ArrayList<>();
		for( int[] a : d )
			for( int b : a )
				l.add(b);
		int[] r = new int[l.size()];
		for( int i = 0; i < l.size(); i++ )
			r[i] = l.get(i);			
		return r;
	}
	
	public static String[] concatenate( String[][] d ) {
		List<String> l = new ArrayList<>();
		for( String[] a : d )
			for( String b : a )
				l.add(b);
		String[] r = new String[l.size()];
		for( int i = 0; i < l.size(); i++ )
			r[i] = l.get(i);			
		return r;
	}

	public static void addConstant(List<double[]> samples, double c) {
		for( int i = 0; i < samples.size(); i++ ) {
			double[] d = samples.get(i);
			double[] nd = Arrays.copyOf(d, d.length+1);
			nd[nd.length-1] = c;
			samples.set(i, nd);
		}	
	}
		
	public static void printSummary(List<double[]> samples, int[] fa  ) {
		for( int i : fa ) {
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for( double[] d : samples )
				ds.addValue(d[i]);
			System.out.println(i+" --> min: "+ds.getMin()+", mean: "+ds.getMean()+", max: "+ds.getMax()+", sd: "+ds.getStandardDeviation());
		}		
	}
	
	public static DoubleMatrix getY(List<double[]> samples, int ta) {
		double[] y = new double[samples.size()];
		for (int i = 0; i < samples.size(); i++)
			y[i] = samples.get(i)[ta];
		return new DoubleMatrix(y);
	}	
	
	public static DoubleMatrix getX(List<double[]> samples, int[] fa, boolean addIntercept) {		
		double[][] x = new double[samples.size()][];
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			
			x[i] = new double[fa.length + (addIntercept ? 1 : 0) ];
			x[i][x[i].length - 1] = 1.0; // gets overwritten if !addIntercept
			for (int j = 0; j < fa.length; j++) {
				x[i][j] = d[fa[j]];
			}
		}
		return new DoubleMatrix(x);
	}
	
	public static List<double[]> getStripped(List<double[]> a, int[] fa) {
		List<double[]> l = new ArrayList<>();
		for( double[] d : a )
			l.add( DataUtils.strip(d, fa) );
		return l;
	}
	
	public static DoubleMatrix getW(List<double[]> a, List<double[]> b, int[] ga ) {
		Dist<double[]> eDist = new EuclideanDist();
		DoubleMatrix W = new DoubleMatrix(a.size(), b.size());
		for (int i = 0; i < a.size(); i++)
			for (int j = 0; j < b.size(); j++)
				W.put(i, j, eDist.dist(
						new double[] { a.get(i)[ga[0]], a.get(i)[ga[1]] }, 
						new double[] { b.get(j)[ga[0]], b.get(j)[ga[1]] }
					));		
		return W;
	}
		
	public static <T> List<T> subset_row( List<T> l, int[] idx ) {
		List<T> r = new ArrayList<>();
		for( int i : idx )
			r.add(l.get(i));
		return r;
	}
	
	public static List<double[]> subset_row( List<double[]> l, List<Integer> li ) {
		return subset_row(l, toIntArray(li) );
	}
	
	public static List<double[]> subset_columns( List<double[]> x, int[] fa ) {
		List<double[]> r = new ArrayList<>();
		for( double[] d : x )
			r.add( DataUtils.strip(d, fa) );
		return r;
	}
	
	public static int[] toIntArray(Collection<Integer> c) {
		int[] j = new int[c.size()];
		int i = 0;
		for (int l : c)
			j[i++] = l;
		return j;
	}
	
	public static double[] toDoubleArray(Collection<Double> c) {
		double[] j = new double[c.size()];
		int i = 0;
		for (double l : c)
			j[i++] = l;
		return j;
	}
		
	public static double[][] transpose(double[][] matrix) {
	    int rows = matrix.length;
	    int cols = matrix[0].length;
	    double[][] transposed = new double[cols][rows];

	    for (int i = 0; i < rows; i++) {
	        for (int j = 0; j < cols; j++) {
	            transposed[j][i] = matrix[i][j];
	        }
	    }
	    return transposed;
	}
}
