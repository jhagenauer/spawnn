package context.space;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.NG;
import spawnn.ng.ContextNG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.Point;

public class SpaceTest {

	private static Logger log = Logger.getLogger(SpaceTest.class);

	public static void main(String[] args) {

		Random r = new Random();
		int T_MAX = 150000;
		int rcpFieldSize = 80;

		File file = new File("data/redcap/Election/election2004.shp");
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(file, new int[] {}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(file);
		Map<double[], Set<double[]>> ctg = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");

		// build dist matrix and add coordinates to samples
		Map<double[], Map<double[], Double>> distMap = new HashMap<double[], Map<double[], Double>>();
		for (int i = 0; i < samples.size(); i++) {
			double[] d = samples.get(i);
			Point p1 = geoms.get(i).getCentroid();

			distMap.put(d, new HashMap<double[], Double>());
			for (double[] nb : ctg.get(d)) {
				int j = samples.indexOf(nb);

				if (i == j)
					continue;

				Point p2 = geoms.get(j).getCentroid();
				distMap.get(d).put(nb, p1.distance(p2));
			}

			// ugly, but easy
			d[0] = p1.getX();
			d[1] = p1.getY();
			
			// normed coordinates
			d[2] = p1.getX();
			d[3] = p1.getY();
		}
		
		final int[] ga = new int[]{0,1};
		final int[] gaNormed = new int[]{2,3};
		
		final int fa = 7;
		final Dist<double[]> fDist = new EuclideanDist( new int[] { fa });
		final Dist<double[]> gDist = new EuclideanDist( ga );
		
		DataUtils.zScoreColumn(samples, fa);
		
		//DataUtils.normalizeGeoColumns(samples, gaNormed );
		DataUtils.zScoreGeoColumns(samples, gaNormed, gDist );
		final Dist<double[]> normedGDist = new EuclideanDist( gaNormed );
					
		// get knns
		Map<double[], List<double[]>> knns = new HashMap<double[], List<double[]>>();
		for (double[] x : samples) {
			List<double[]> sub = new ArrayList<double[]>();
			while (sub.size() <= rcpFieldSize) { // sub.size() must be larger than cLength!

				double[] minD = null;
				for (double[] d : samples)
					if (!sub.contains(d) && (minD == null || gDist.dist(d, x) < gDist.dist(minD, x)))
						minD = d;
				sub.add(minD);
			}
			knns.put(x, sub);
		}
		log.debug("knn build.");
				
		Map<String,double[]> series = new HashMap<String,double[]>();

		{ // weighted ng
			for (double w : new double[]{ 0.8 } ) {
				log.debug("weighted ng " + w);

				Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
				map.put(fDist, 1-w);
				map.put(normedGDist, w);
				Dist<double[]> wDist = new WeightedDist<double[]>(map);
				
				DefaultSorter<double[]> bg = new DefaultSorter<double[]>(wDist);
							
				NG ng = new NG(100, 50, 0.01, 0.5, 0.005, samples.get(0).length, bg);

				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get( r.nextInt(samples.size()));
					ng.train((double) t / T_MAX, x);			
				}

				Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
				double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);		
				series.put("WNG " + w, qe);
				
				DataUtils.writeCSV("output/wng_neurons_"+w+".csv", ng.getNeurons(), null);
				Drawer.geoDrawCluster(bmus.values(), samples, geoms, "output/wng_"+w+".png", true);
			}
		}
		
		{ // geosom
			for( int l : new int[]{ 1 } ) {
				log.debug("geosom "+l);
				
				spawnn.som.bmu.BmuGetter<double[]> bg = new spawnn.som.bmu.KangasBmuGetter<double[]>( normedGDist, fDist, l );
				Grid2D<double[]> grid = new Grid2DHex<double[]>(10,10);
				SomUtils.initRandom(grid, samples);
						
				SOM som = new SOM( new GaussKernel( new LinearDecay(10, 1)), new LinearDecay(1.0,0.0), grid, bg );
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get( r.nextInt(samples.size()));
					som.train((double) t / T_MAX, x);
				}

				Map<GridPos, Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bg);
				double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
				series.put("GeoSOM " + l, qe);
				
				DataUtils.writeCSV("output/geosom_neurons_"+l+".csv", grid.getPrototypes(), null);
				Drawer.geoDrawCluster(bmus.values(), samples, geoms, "output/GeoSOM_"+l+".png", true);
			}
		}
		
		{ // cng
			for( int l : new int[]{ 2 } ) {
				log.debug("cng: "+l);
				
				KangasSorter<double[]> bg = new KangasSorter<double[]>( normedGDist, fDist, l );
				NG ng = new NG(100, 50.0, 0.01, 0.5, 0.005, samples.get(0).length, bg);

				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get( r.nextInt(samples.size()));
					ng.train((double) t / T_MAX, x);			
				}

				Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
				double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
				series.put("CNG " + l, qe);
				
				DataUtils.writeCSV("output/cng_neurons.csv", ng.getNeurons(), null);
				Drawer.geoDrawCluster(bmus.values(), samples, geoms, "output/cng_"+l+".png", true);
			}
		}
		
		{ // WMDMNG
			List<double[]> settings = new ArrayList<double[]>();
			settings.add( new double[]{ 4, 155000, 0.6, 0.5 } );
			settings.add( new double[]{ 4, -1, 0.75, 0.6 } );
			
			settings.add( new double[]{ 4, -1, 0.8, 0.7 } );
			settings.add( new double[]{ 4, -1, 0.8, 0.8 } );
			settings.add( new double[]{ 4, -1, 0.75, 0.5 } );
			settings.add( new double[]{ 4, -1, 0.8, 0.6 } );
			
			for( double[] s : settings ) {
				int numDir = (int)s[0];
				double band = s[1];
				double alpha = s[2];
				double beta = s[3];
								
				log.debug("WMDMNG " + alpha+","+beta+","+band+","+numDir);
							
				final Map<double[], Map<double[], Double>> dMap;
				if( band < 0 ) {
					dMap = new HashMap<double[],Map<double[],Double>>();
					for( double[] d : ctg.keySet() ) {
						Map<double[],Double> dists = new HashMap<double[],Double>();
						for( double[] nb : ctg.get(d) )
							dists.put( nb, 1.0 );
						
						double n = dists.size();
						for( double[] nb : ctg.get(d) )
							dists.put( nb, 1.0/n );
						
						dMap.put(d, dists );
					}
				} else {
					dMap = SpaceTest.getDistMatrix(samples, gDist, band );
				}
							
				List<double[]> neurons = new ArrayList<double[]>();
				for( int i = 0; i < 100; i++ ) {
					double[] rs = samples.get(r.nextInt(samples.size()));
					double[] d = Arrays.copyOf(rs, rs.length * (numDir+1) );
					for( int j = rs.length; j < d.length; j++ )
						d[j] = r.nextDouble();
					neurons.add( d );
				}
				
				Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
				for (double[] d : samples)
					bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));
						
				SorterWMC bg = new SorterWMC(bmuHist, dMap, fDist, alpha, beta );
				ContextNG ng = new ContextNG(neurons, (double)neurons.size()/2, 0.01, 0.5, 0.005, bg);

				bg.bmuHistMutable = true;
				for (int t = 0; t < T_MAX; t++) {
					double[] x = samples.get( r.nextInt(samples.size()));
					ng.train((double) t / T_MAX, x);			
				}
				bg.bmuHistMutable = false;

				Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), bg);
				double[] qe = SpaceTest.getQuantizationErrorOld(samples, bmus, fDist, rcpFieldSize, knns);
				series.put("WMDMNG " + alpha+","+beta+","+band+","+numDir,qe);
				
				DataUtils.writeCSV("output/wmdng_neurons.csv", ng.getNeurons(), null);
				Drawer.geoDrawCluster(bmus.values(), samples, geoms, "output/wmdmng.png", true);
				
				/*Map <double[],Set<List<double[]>>> rf = getRFs(knns, bmus);
				for( double[] bmu : rf.keySet() ) {	
					
					List<double[]> l = new ArrayList<double[]>();
					for( int i = 0; i < rcpFieldSize; i++ ) {	
						SummaryStatistics ss = new SummaryStatistics();
						for( List<double[]> s : rf.get(bmu) ) 
							ss.addValue(s.get(i)[fa]);
						
						l.add( new double[]{ss.getMean(), ss.getMean()-ss.getStandardDeviation(), ss.getMean()+ss.getStandardDeviation()} );
					}
					
					DecimalFormat df = new DecimalFormat("000");
					BufferedWriter bw = null;
					try {
						bw = new BufferedWriter( new FileWriter("output/wmdmng_"+bmu.hashCode()+"_"+df.format(rf.get(bmu).size())+".csv"));
						bw.write("m,l,u\n");					
						for( int i = 0; i < rcpFieldSize; i++ ) 
							for( double[] d : l )
								bw.write(d[0]+","+d[1]+","+d[2]+"\n");
					} catch (IOException e1) {
						e1.printStackTrace();
					} finally {
						try { bw.close(); } catch( Exception e ) {}
					}	
				}*/
			}
		}
		
		// export csv
		BufferedWriter bw = null;
		try {
			bw = new BufferedWriter( new FileWriter("output/election_results.csv"));
			StringBuffer sb = new StringBuffer();
			for( String s : series.keySet() )
				sb.append("\""+s+"\",");
			sb.deleteCharAt(sb.lastIndexOf(","));
			bw.write(sb.toString()+"\n");
			
			for( int i = 0; i < rcpFieldSize; i++ ) {
				sb = new StringBuffer();
				for( String s : series.keySet() ) {
					double d = series.get(s)[i];
					sb.append(d+",");
				}
				sb.deleteCharAt(sb.lastIndexOf(","));
				bw.write(sb.toString()+"\n");
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		} finally {
			try { bw.close(); } catch( Exception e ) {}
		}
		
		
		XYSeriesCollection errors = new XYSeriesCollection();
		for( String s : series.keySet() ) {
			double[] qe = series.get(s);
			XYSeries error = new XYSeries(s);
			for (int size = 0; size < qe.length; size++)
				error.add(size, qe[size]);
			errors.addSeries(error);
		}			
		JFreeChart lineplot = ChartFactory.createXYLineChart("lineplot", "knn", "Error", errors, PlotOrientation.VERTICAL, true, true, false);
		try {
			ChartUtilities.saveChartAsPNG(new File("output/lineplot_space.png"), lineplot, 800*2, 600*2);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	// if bmu does not map any data, it has not a receptive field
	public static <T> Map<T, Set<List<double[]>>> getRFs(  Map<double[], List<double[]>> knn, Map<T, Set<double[]>> bmuMapping ) {
		Map<T, Set<List<double[]>>> bmuSeqs = new HashMap<T, Set<List<double[]>>>(); 
		for( T bmu : bmuMapping.keySet() ) {
			for( double[] x : bmuMapping.get(bmu) ) {
				if (!bmuSeqs.containsKey(bmu))
					bmuSeqs.put(bmu, new HashSet<List<double[]>>());
				bmuSeqs.get(bmu).add(knn.get(x));
			}
		}
		return bmuSeqs;
	}
	
	public static <T> Map<T, List<double[]>> getMeanRF( Map<T, Set<List<double[]>>> rf ) {
		int rfSize = rf.values().iterator().next().iterator().next().size();
		int dSize = rf.values().iterator().next().iterator().next().get(0).length;
		
		Map<T, List<double[]>> meanRFs = new HashMap<T, List<double[]>>();
				
		for (T bmu : rf.keySet()) {

			List<double[]> meanList = new ArrayList<double[]>();
			for (int i = 0; i < rfSize; i++) {
				double[] d = new double[dSize];
				
				Set<List<double[]>> s = rf.get(bmu);
				for (List<double[]> l : s)
					for (int j = 0; j < d.length; j++)
						d[j] += l.get(i)[j] / s.size();

				meanList.add(d);
			}

			meanRFs.put(bmu, meanList);
		}
		return meanRFs;
	}

	@Deprecated
	public static <T> double[] getQuantizationErrorOld(List<double[]> samples, Map<T, Set<double[]>> bmus, Dist<double[]> fDist, int rcpFieldSize, Map<double[], List<double[]>> knns) {
		Map<T, Set<List<double[]>>> bmuSeqs = getRFs(knns, bmus);
		Map<T, List<double[]>> meanRFs = getMeanRF(bmuSeqs);
		
		double[] qe = new double[rcpFieldSize];
		for (int k = 0; k < rcpFieldSize; k++) {
			int num = 0; // = samples.size()
			double sum = 0;				
			for( T bmu : bmus.keySet() ) {
				double[] mrf = meanRFs.get(bmu).get(k);
				for( double[] x : bmus.get(bmu) ) {
					sum += Math.pow( fDist.dist( knns.get(x).get(k), mrf ), 2);
					num++;
				}
					
			}
			qe[k] = Math.sqrt(sum / num); // average error TODO: sqrt, wirklich???
		}
		return qe;
	}
	
	public static <T> double[] getQuantizationError(List<double[]> samples, Map<T, Set<double[]>> bmuMapping, Dist<double[]> fDist, int rcpFieldSize, Map<double[], List<double[]>> knns) {
		Map<T, Set<List<double[]>>> bmuSeqs = getRFs(knns, bmuMapping);
		Map<T, List<double[]>> meanRFs = getMeanRF(bmuSeqs);
		
		double[] qe = new double[rcpFieldSize];
		for (int k = 0; k < rcpFieldSize; k++) {
			double sum = 0;				
			for( T bmu : meanRFs.keySet() ) {
				double[] mrf = meanRFs.get(bmu).get(k);
				for( double[] x : bmuMapping.get(bmu) )
					sum += Math.pow( fDist.dist( knns.get(x).get(k), mrf ), 2);
			}
			qe[k] = sum; 
		}
		return qe;
	}
	
	public static Map<double[], Map<double[], Double>> getDistMatrix(List<double[]> samples, Dist<double[]> gDist, double cutOff ) {
			
		Map<double[], Map<double[], Double>> dMap = new HashMap<double[], Map<double[], Double>>();
		for (double[] d : samples) {

			// get inverse dists
			Map<double[], Double> dists = new HashMap<double[], Double>();
			for (double[] nb : samples) {
				double gd = gDist.dist(d, nb);

				if( d == nb || gd > cutOff )
					continue;
				
				dists.put(nb, cutOff - gd );
			}
			
			if( dists.isEmpty() )
				log.warn("Sample with no neighbors!!!");
		
			double sum = 0;
			for (double dist : dists.values())
				sum += dist;

			// row-normalize
			Map<double[], Double> nDists = new HashMap<double[], Double>();
			for (double[] nb : dists.keySet())
				nDists.put(nb, dists.get(nb) / sum);

			dMap.put(d, dists);
		}
		return dMap;
	}
} 
