package houseprice;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;
import org.geotools.data.simple.SimpleFeatureIterator;
import org.geotools.data.simple.SimpleFeatureSource;
import org.geotools.geometry.jts.ReferencedEnvelope;
import org.geotools.grid.Grids;
import org.opengis.feature.Feature;
import org.opengis.feature.simple.SimpleFeature;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.utils.NGUtils;
import spawnn.utils.Clustering;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.SpatialDataFrame;
import spawnn.utils.Clustering.HierarchicalClusteringType;
import spawnn.utils.Clustering.TreeNode;

public class SubmarketsByClusterLM {
	private static Logger log = Logger.getLogger(SubmarketsByClusterLM.class);

	public static <GridFeatureBuilder> void main(String[] args) {
		final GeometryFactory gf = new GeometryFactory();
		final Random r = new Random();

		final List<double[]> samples = new ArrayList<double[]>();
		final List<double[]> desired = new ArrayList<double[]>();
	
		final int[] ga = new int[] { 0, 1 };
		final SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromCSV(new File("output/houseprice.csv"), ga, new int[] {}, true);
		for (double[] d : sdf.samples) {
			double[] nd = Arrays.copyOf(d, d.length - 1);
			samples.add(nd);
			desired.add(new double[] { d[d.length - 1] });
		}
		final List<Geometry> geoms = sdf.geoms;

		final int[] fa = new int[samples.get(0).length - 2]; // omit geo-vars
		for (int i = 0; i < fa.length; i++)
			fa[i] = i + 2;

		final Dist<double[]> gDist = new EuclideanDist(ga);
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		final SpatialDataFrame bez = DataUtils.readSpatialDataFrameFromShapefile(new File("data/marco/dat4/bez_2008.shp"), true);
		Geometry ab = null;
		for( Geometry g : bez.geoms ) {
			if( ab == null )
				ab = g;
			else
				ab = ab.union(g);
		}
		
		final Envelope abEnv = ab.getEnvelopeInternal();
	    ReferencedEnvelope gridBounds = new ReferencedEnvelope(abEnv,null);
	    SimpleFeatureSource grid = Grids.createHexagonalGrid(gridBounds, 5000);
	    final List<Geometry> gridGeoms = new ArrayList<Geometry>();
	    SimpleFeatureIterator it;
		try {
			it = grid.getFeatures().features();
			while( it.hasNext() ) {
				SimpleFeature sf = (SimpleFeature)it.next();
				Geometry g = (Geometry)sf.getDefaultGeometry();
				if( g.intersects(ab) )
					gridGeoms.add(g);
			}
			it.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
				

		//String srsHouseprice = "PROJCS[\"unnamed\",GEOGCS[\"Bessel 1841\",DATUM[\"unknown\",SPHEROID[\"bessel\",6377397.155,299.1528128]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"standard_parallel_1\",46],PARAMETER[\"standard_parallel_2\",49],PARAMETER[\"latitude_of_origin\",48],PARAMETER[\"central_meridian\",13.3333333333333],PARAMETER[\"false_easting\",400000],PARAMETER[\"false_northing\",400000],UNIT[\"Meter\",1]]";
		log.debug("Starting optimization...");
		ExecutorService es = Executors.newFixedThreadPool(4);
		List<Future<double[]>> futures = new ArrayList<Future<double[]>>();

		for (int run = 0; run < 1; run++) {

			int samplesSize = samples.size();
			final List<double[]> samplesTrain = new ArrayList<double[]>(samples);
			final List<double[]> desiredTrain = new ArrayList<double[]>(desired);

			final List<double[]> samplesVal = new ArrayList<double[]>();
			final List<double[]> desiredVal = new ArrayList<double[]>();
			while (samplesVal.size() < 0.3 * samplesSize) {
				int idx = r.nextInt(samplesTrain.size());
				samplesVal.add(samplesTrain.remove(idx));
				desiredVal.add(desiredTrain.remove(idx));
			}

			futures.add(es.submit(new Callable<double[]>() {
				@Override
				public double[] call() throws Exception {
					// split training data again 
					List<double[]> subSamplesTrain = new ArrayList<double[]>(samplesTrain);
					List<double[]> subDesiredTrain = new ArrayList<double[]>(desiredTrain);

					List<double[]> subSamplesVal = new ArrayList<double[]>();
					List<double[]> subDesiredVal = new ArrayList<double[]>();
					while (subSamplesVal.size() < 0.3 * samplesTrain.size() ) {
						int idx = r.nextInt(subSamplesTrain.size());
						subSamplesVal.add(subSamplesTrain.remove(idx));
						subDesiredVal.add(subDesiredTrain.remove(idx));
					}
					
					Map<Geometry,Set<double[]>> cellObsMap = new HashMap<Geometry,Set<double[]>>();
					for( Geometry cell : gridGeoms ) {
						Set<double[]> s = new HashSet<double[]>();
						for( double[] d : subSamplesTrain ) {
							int idx = samples.indexOf(d);
							Geometry p = geoms.get(idx);
							if( cell.intersects(p) )
								s.add(d);
						}
						cellObsMap.put(cell,s);
					}
					
					// idee: zentroiden verteilen (zufällig), leere raus löschen, ward clustern
					// random centroids
					List<Geometry> centroids = new ArrayList<Geometry>();
					while( centroids.size() < 100 ) {
						double x = r.nextDouble() * abEnv.getWidth() + abEnv.getMinX();
						double y = r.nextDouble() * abEnv.getHeight() + abEnv.getMinY();
						Geometry point = gf.createPoint(new Coordinate(x,y));
						for( Geometry cell : cellObsMap.keySet() )
							if( cell.intersects(point))
								centroids.add(point);
					}
					log.debug("Pre-cluster: "+centroids.size() );
					
					// build mapping
					Map<Geometry,Set<Geometry>> centCellMap;
					while( true ) {
						centCellMap = new HashMap<Geometry,Set<Geometry>>();
						for( Geometry cell : cellObsMap.keySet() ) {
							Geometry closestCent = null;
							for( Geometry p : centroids ) 
								if( closestCent == null || p.distance(cell) < closestCent.distance(cell) )
									closestCent = p;
							if( !centCellMap.containsKey(closestCent) )
								centCellMap.put(closestCent, new HashSet<Geometry>() );
							centCellMap.get(closestCent).add(cell);
						}
						
						// remove a random empty cluster
						List<Geometry> emtyCentroids = new ArrayList<Geometry>(centroids);
						for( Entry<Geometry, Set<Geometry>> e : centCellMap.entrySet() ) {
							for( Geometry cell : e.getValue() )
								if( !cellObsMap.get(cell).isEmpty() )
									emtyCentroids.remove(e.getKey() );
						}
						if( !emtyCentroids.isEmpty() ) {
							Geometry e = emtyCentroids.get(r.nextInt(emtyCentroids.size()));
							centroids.remove(e);
						} else
							break;
					}
					log.debug("After deletion of empty cluster: "+centCellMap.size());
					
					//Clustering.getHierarchicalClusterTree(cm, dist, Clustering.HierarchicalClusteringType.ward);
										
					return new double[] {};
				}
			}));
		}
 		es.shutdown();

		DescriptiveStatistics ds[] = null;
		for (Future<double[]> ff : futures) {
			try {
				double[] ee = ff.get();
				if (ds == null) {
					ds = new DescriptiveStatistics[ee.length];
					for (int i = 0; i < ee.length; i++)
						ds[i] = new DescriptiveStatistics();
				}
				for (int i = 0; i < ee.length; i++)
					ds[i].addValue(ee[i]);
			} catch (InterruptedException ex) {
				ex.printStackTrace();
			} catch (ExecutionException ex) {
				ex.printStackTrace();
			}
		}
	}
}
