package spawnn.som.bmu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;

import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.Drawer;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;

public class GWBmuGetter extends BmuGetter<double[]> {
	
	private static Logger log = Logger.getLogger(GWBmuGetter.class);

	private Dist<double[]> fDist, gDist;
	private GWKernel kernel;
	private Map<double[], Double> bandwidth;
	
	public GWBmuGetter(Dist<double[]> gDist, Dist<double[]> fDist, GWKernel k, Map<double[], Double> bandwidth) {
		this.gDist = gDist;
		this.fDist = fDist;
		this.kernel = k;
		this.bandwidth = bandwidth;
	}

	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		double dist = Double.NaN;
		GridPos bmu = null;
		
		for (GridPos p : grid.getPositions()) {

			if (ign != null && ign.contains(p))
				continue;
			
			double[] v = grid.getPrototypeAt(p);
			double w = GeoUtils.getKernelValue( kernel, gDist.dist(v, x), bandwidth.get(x) );
			double[] est = new double[v.length];
			for( int i = 0; i < v.length; i++ )
				est[i] = w * v[i];
			double d = fDist.dist(est, x);
			
			if( bmu == null || d < dist ) { 
				bmu = p;
				dist = d;
			}
		}
		return bmu;
	}

	public static void main(String[] args) {
		Random r = new Random();
		int T_MAX = 100000;

		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}
		List<double[]> samples = sdf.samples;

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 21, 32, 54, 63, 49 };*/
		
		GeometryFactory gf = new GeometryFactory();
		List<double[]> samples = new ArrayList<>();
		List<Geometry> geoms = new ArrayList<>();
		Map<double[],Integer> classes = new HashMap<>();
		while( samples.size() < 500 ) {
			double x = r.nextDouble();
			double y = r.nextDouble();
			double z;
			int c;
			double noise = r.nextDouble()*0.1-0.05;
			if( x < 0.3  ) {
				z=1;
				c=0;
			} else if ( x > 0.6 ) {
				z = 1;
				c=1;
			} else {
				z=0;
				c=2;
			}
			double[] d = new double[]{x,y,z+noise};
			classes.put(d, c);
			samples.add( d );
			geoms.add( gf.createPoint( new Coordinate(x,y)));
		}
		
		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 2 };
		
		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		DataUtils.transform(samples, fa, Transform.zScore);
		//DataUtils.writeCSV("output/squares.csv", samples,new String[]{"x","y","z"});
			
		List<Entry<List<Integer>, List<Integer>>> cvList = SupervisedUtils.getCVList(2, 10, samples.size());
				
		boolean adaptive = false;		
		
		for( GWKernel ke : new GWKernel[]{ GWKernel.gaussian } )
		for( double bw : new double[]{ 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2 } ) {
			
			SummaryStatistics ss = new SummaryStatistics();
			for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
				List<double[]> samplesTrain = new ArrayList<double[]>();
				for( int k : cvEntry.getKey() ) 
					samplesTrain.add(samples.get(k));
								
				List<double[]> samplesVal = new ArrayList<double[]>();
				for( int k : cvEntry.getValue() ) 
					samplesVal.add(samples.get(k));
				
				Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samplesVal, gDist, bw, adaptive);
				double d = 0;
				for( double[] uv : samplesVal ) 
					d += Math.pow(fDist.dist(uv, GeoUtils.getGWMean(samplesTrain, uv, gDist, ke, bandwidth.get(uv) ) ),2);
				ss.addValue(d);
			}			
			
			Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
			List<double[]> values = new ArrayList<>();
			double d = 0;
			for( double[] x : samples ) {
				double[] v = GeoUtils.getGWMean(samples, x, gDist, ke, bandwidth.get(x) );
				values.add( v );
				d += Math.pow(fDist.dist(v, x),2);
			}
			
			Drawer.geoDrawValues(geoms, values, fa[0], null, ColorBrewer.Blues, "output/gwmean_"+ke+"_"+bw+".png");
			//DataUtils.writeCSV("output/gwmean_"+ke+"_"+bw+".csv", values,new String[]{"x","y","z"});
			
			log.debug(ke+", "+bw+","+ss.getMean()+","+(d/samples.size()));
		}
		
		log.debug("GWSOM");
		//for( GWKernel ke : GWKernel.values() )
		for( GWKernel ke : new GWKernel[]{ GWKernel.gaussian } )
		for (double bw : new double[] { 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }) { // fqe sinkt, sqe steigt mit steigender bw, sudden increase of sqe below 0.03.. why?
			
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			
			Map<double[], Double> bandwidth = GeoUtils.getBandwidth(samples, gDist, bw, adaptive);
			BmuGetter<double[]> bmuGetter = new GWBmuGetter(gDist, fDist, ke, bandwidth);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1 )), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			log.debug(ke+","+ bw);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
			
			Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
			SomUtils.printGeoGrid(ga, grid, "output/gwsom_grid_"+ke+"_"+bw+".png");
			SomUtils.printDMatrix(grid, fDist, "output/gwsom_dmat_"+ke+"_"+bw+".png");
			SomUtils.printClassDist(classes, mapping, grid, "output/gwsom_class_"+ke+"_"+bw+".png");
		}
		
		log.debug("GeoSOM");
		for (int k : new int[]{ 1,2,3 }) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, k);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			log.debug("k: " + k);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
			
			Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
			SomUtils.printGeoGrid(ga, grid, "output/geosom_grid_"+k+".png");			
			SomUtils.printDMatrix(grid, fDist, "output/geosom_dmat_"+k+".png");
			SomUtils.printClassDist(classes, mapping, grid, "output/geosom_class_"+k+".png");
			
		}
		
		log.debug("WeightedSOM");
		for (double w : new double[]{}) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			
			Map<Dist<double[]>,Double> m = new HashMap<>();
			m.put(gDist, w);
			m.put(fDist, 1.0-w);
			Dist<double[]> wDist = new WeightedDist<>(m);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<>(wDist);

			SOM som = new SOM(new GaussKernel(new LinearDecay(10, 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}

			log.debug("w: " + w);
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
			
			Map<GridPos,Set<double[]>> mapping = SomUtils.getBmuMapping(samples, grid, bmuGetter,true);
			SomUtils.printGeoGrid(ga, grid, "output/weightedsom_grid_"+w+".png");
			SomUtils.printDMatrix(grid, fDist, "output/weightedsom_dmat_"+w+".png");
			SomUtils.printClassDist(classes, mapping, grid, "output/weightedsom_class_"+w+".png");
			
		}
	}
}
