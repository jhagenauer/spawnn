package gwsom;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.Point;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.GWBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ClusterValidation;
import spawnn.utils.Clustering;
import spawnn.utils.Clustering.TreeNode;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class GWSOM_Squares {

	private static Logger log = Logger.getLogger(GWSOM_Squares.class);
	
	enum tests { squares, regions, census };

	public static void main(String[] args) {
		
		for( tests t : tests.values() ) {

			GeometryFactory gf = new GeometryFactory();
			Random r = new Random();
			int T_MAX = 100000;
			
			List<double[]> samples;
			List<Geometry> geoms;
						
			int[] ga;
			int[] fa;
			
			if( t == tests.squares ) {
				samples = new ArrayList<>();
				geoms = new ArrayList<>();
				while( samples.size() < 1000 ) {
					double x = r.nextDouble();
					double y = r.nextDouble();
					double z;
					double noise = r.nextDouble()*0.1-0.05;
					int c = 0;
					if( x < 0.33  ) {
						z=1;
						c = 1;
					} else if ( x > 0.66 ) {
						z=1;
						c = 2;
					} else {
						z=0;
						c = 3;
					}
					double[] d = new double[]{x,y,z+noise,c};
					samples.add( d );
					geoms.add( gf.createPoint( new Coordinate(x,y)));
				}
				
				ga = new int[] { 0, 1 };
				fa = new int[] { 2 };
			} else if( t == tests.regions ) {
				SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/100regions.shp"), true);
				samples = sdf.samples;
				geoms = sdf.geoms;
				
				ga = new int[] { 0, 1 };
				fa = new int[] { 2 };
			} else {
				SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
				for (int i = 0; i < sdf.samples.size(); i++) {
					Point p = sdf.geoms.get(i).getCentroid();
					sdf.samples.get(i)[0] = p.getX();
					sdf.samples.get(i)[1] = p.getY();
				}
				samples = sdf.samples;
				geoms = sdf.geoms;

				ga = new int[] { 0, 1 };
				fa = new int[] { 11, 21, 23, 32, 54, 63, 48, 49 };
			}
						
			Map<Integer,Set<double[]>> cm = new HashMap<>();
			for( double[] d : samples ) {
				int c = (int)d[3];
				if( !cm.containsKey(c))
					cm.put(c, new HashSet<double[]>() );
				cm.get(c).add(d);
			}
			
			DataUtils.transform(samples, fa, Transform.zScore);
			
			Dist<double[]> fDist = new EuclideanDist(fa);
			Dist<double[]> gDist = new EuclideanDist(ga);
			
			double maxDist = 0;
			for( int i = 0; i < samples.size()-1; i++ )
				for( int j = i+1; j < samples.size(); j++ )
					maxDist = Math.max(maxDist, gDist.dist( samples.get(i), samples.get(j) ) );
							
			boolean adaptive = false;
			int runs = 8;
			int threads = 4;
						
			Path file = Paths.get("output/results_"+t+"_gwbmu.csv");
			try {
				Files.createDirectories(file.getParent());
				Files.deleteIfExists(file);
				Files.createFile(file);
				String s = "method,parameter,variable,value\r\n";
				Files.write(file, s.getBytes(), StandardOpenOption.APPEND);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			
			log.debug("GWSOM");
			for( final GWKernel ke : new GWKernel[]{ GWKernel.gaussian, GWKernel.boxcar, GWKernel.bisquare } )
			for (double bw = 0.01; bw <= maxDist*2; bw+=maxDist/200 ) {
				final double BW = bw;
				
				ExecutorService es = Executors.newFixedThreadPool(threads);		
				for( int i = 0; i < runs; i++ )
					es.submit(new Runnable() {
						@Override
						public void run() {
							Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
							SomUtils.initRandom(grid, samples);
							
							Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, BW, adaptive);
							BmuGetter<double[]> bmuGetter = new GWBmuGetter(gDist, fDist, ke, bandwidth);
							
							SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1 )), new LinearDecay(1.0, 0.0), grid, bmuGetter);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								som.train((double) t / T_MAX, x);
							}
	
							synchronized(this) {
								try {
									Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
									Files.write(file, (ke+"," + BW + ",fqe," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, (ke+"," + BW + ",sqe," + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, (ke+"," + BW + ",te," + SomUtils.getTopoError(grid, bmuGetter, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, (ke+"," + BW + ",nmi," + (ClusterValidation.getNormalizedMutualInformation(getCluster(grid, fDist, mapping, cm.size()), cm.values() ) )+"\r\n" ).getBytes(), StandardOpenOption.APPEND);
								} catch (IOException e) {
									e.printStackTrace();
								}
							}
						}
					});		
				es.shutdown();
				try { es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); } catch (InterruptedException e) { e.printStackTrace(); }
			}
			
			log.debug("GeoSOM");
			for (int k = 1; k <= 26; k++ ) {
				final int K = k;
				
				ExecutorService es = Executors.newFixedThreadPool(threads);
				for( int i = 0; i < runs; i++ )
					es.submit(new Runnable() {
						@Override
						public void run() {
							Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
							SomUtils.initRandom(grid, samples);
							BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, K);
	
							SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								som.train((double) t / T_MAX, x);
							}
							
							synchronized (this) {
								try {
									Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
									Files.write(file, ("geosom," + K + ",fqe," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, ("geosom," + K + ",sqe," + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, ("geosom," + K + ",te," + SomUtils.getTopoError(grid, bmuGetter, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
									Files.write(file, ("geosom," + K + ",nmi," + (ClusterValidation.getNormalizedMutualInformation(getCluster(grid, fDist, mapping, cm.size()), cm.values() ) )+"\r\n" ).getBytes(), StandardOpenOption.APPEND);
								} catch (IOException e) {
									e.printStackTrace();
								}		
							}							
						}
					});	
				es.shutdown();
				try { es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); } catch (InterruptedException e) { e.printStackTrace(); }
			}
			
			log.debug("WeightedSOM");
			for (double w = 0; w <= 1.0; w+=0.01 ) {
				final double W = w;
				
				ExecutorService es = Executors.newFixedThreadPool(threads);
				for( int i = 0; i < runs; i++ )
					es.submit(new Runnable() {
						
						@Override
						public void run() {
							Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
							SomUtils.initRandom(grid, samples);
							
							Map<Dist<double[]>,Double> m = new HashMap<>();
							m.put(gDist, 1.0-W);
							m.put(fDist, W);
							Dist<double[]> wDist = new WeightedDist<>(m);
							BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<>(wDist);
	
							SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
							for (int t = 0; t < T_MAX; t++) {
								double[] x = samples.get(r.nextInt(samples.size()));
								som.train((double) t / T_MAX, x);
							}
							
							try {
								Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
								Files.write(file, ("weighted," + W + ",fqe," + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
								Files.write(file, ("weighted," + W + ",sqe," + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
								Files.write(file, ("weighted," + W + ",te," + SomUtils.getTopoError(grid, bmuGetter, samples)+"\r\n").getBytes(), StandardOpenOption.APPEND);
								Files.write(file, ("weighted," + W + ",nmi," + (ClusterValidation.getNormalizedMutualInformation( getCluster(grid, fDist, mapping, cm.size()), cm.values() ) )+"\r\n" ).getBytes(), StandardOpenOption.APPEND);
							} catch (IOException e) {
								e.printStackTrace();
							}						
						}
					});	
				es.shutdown();
				try { es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS); } catch (InterruptedException e) { e.printStackTrace(); }
			}
		}
	}

	public static Collection<Set<double[]>> getCluster(Grid<double[]> grid, Dist<double[]> dist, Map<GridPos, Set<double[]>> mapping, int nrCluster ) {
		Map<double[], Set<double[]>> cm = new HashMap<double[], Set<double[]>>();
		for (GridPos p : grid.getPositions()) {
			double[] v = grid.getPrototypeAt(p);
			Set<double[]> s = new HashSet<double[]>();
			for (GridPos nb : grid.getNeighbours(p))
				s.add(grid.getPrototypeAt(nb));
			cm.put(v, s);
		}
		
		List<TreeNode> tree = Clustering.getHierarchicalClusterTree(cm, dist, Clustering.HierarchicalClusteringType.ward );
		List<Set<double[]>> clusters = Clustering.treeToCluster( Clustering.cutTree( tree, nrCluster ) );
		
		List<Set<GridPos>> gpCluster = new ArrayList<>();
		for( Set<double[]> s : clusters ) {
			Set<GridPos> ns = new HashSet<>();
			for( double[] d : s )
				ns.add( grid.getPositionOf(d));
			gpCluster.add(ns);
		}
		
		Map<double[], Set<double[]>> ll = new HashMap<double[], Set<double[]>>();
		for (Set<GridPos> s : gpCluster) { // for each cluster s
			Set<double[]> l = new HashSet<double[]>();
			for (GridPos p : s) // for each cluster element p
				if (mapping.containsKey(p))
					l.addAll(mapping.get(p));
			if (!l.isEmpty())
				ll.put(DataUtils.getMean(l), l);
		}
		return ll.values();
	}
}
