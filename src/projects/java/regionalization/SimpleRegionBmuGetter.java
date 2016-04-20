package regionalization;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Geometry;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

/* TODO:
 * - Konvergenzuntersuchung
 * - Filmchen
 * - Test unterschiedliche map-größen oder topologien
 * - effekt unterschieldicher initialisierungen ( mit finit besser )
 * - warum nicht einfach andere regionalisierung, und dann in som packen. dazu müßte man die enstandenen regionen als einzelne vektoren dartellen
 * - interessant wär ees die nachbarschaft von raum und attributen zu vergleichen
 */

public class SimpleRegionBmuGetter extends BmuGetter<double[]> {

	private static Logger log = Logger.getLogger(SimpleRegionBmuGetter.class);

	private Dist<double[]> fDist;
	private Map<double[], Set<double[]>> cm;
	public Map<GridPos, Set<double[]>> map; // mapping
	public Map<GridPos, Integer> nonCont; // non contiguos grid positions

	public SimpleRegionBmuGetter(Map<double[], Set<double[]>> cm, Dist<double[]> fDist) {
		this.fDist = fDist;
		this.cm = cm;
		this.map = new HashMap<GridPos, Set<double[]>>();
		this.nonCont = new HashMap<GridPos, Integer>();
	}

	@Override
	public GridPos getBmuPos(double[] x, Grid<double[]> grid, Set<GridPos> ign) {
		
		// update gridpos
		for( GridPos p : grid.getPositions() )
			if( !map.containsKey(p) )
				map.put( p, new HashSet<double[]>() );
		map.keySet().retainAll( grid.getPositions() );
				
		// remove old mapping of x 
		for( GridPos p : grid.getPositions() ) {
			if( map.get(p).contains(x)) {
				map.get(p).remove(x);
				
				// if we cant remove p, keep it and return
				if( !RegionUtils.isContiugous(cm, map.get(p) ) ) {
					map.get(p).add(x);
					return p;
				}
				break;
			}
		}

		// get grid positions that are neighboring x, so that adding results in contiguous mappings
		Set<GridPos> cg = new HashSet<GridPos>();
		for (GridPos p : grid.getPositions()) {

			if( map.get(p).isEmpty())
				cg.add(p);

			for (double[] d : map.get(p)) {
				if( cm.get(d).contains(x) ) { 
					cg.add(p);
					continue;
				}
			}
		}
		
		if( cg.isEmpty() ) { // does not happen
			log.debug("empty");
			// IDEA: Dont remove smallest, but worst
			GridPos smallest = null;
			for( GridPos p : map.keySet() )
				if( smallest == null || map.get(p).size() < map.get(smallest).size() )
					smallest = p;
			map.get(smallest).clear();
			cg.add(smallest);
		}
								
		GridPos bmu = null;	
		double dist = Double.POSITIVE_INFINITY;
		for (GridPos p : cg) {
			double[] v = grid.getPrototypeAt(p);
			double d = fDist.dist(v, x);
			if (d < dist) {
				dist = d;
				bmu = p;
			}
		}
				
		map.get(bmu).add(x);
				
		return bmu;
	}

	public Map<GridPos, Set<double[]>> getMap() {
		return map;
	}
	
	public void setMap( Map<GridPos, Set<double[]>> map ) {
		this.map = map;
	}

	public static void main(String[] args) {
		final Random r = new Random();

		final int T_MAX = 500000;

		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/200rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);
		//List<Geometry> geoms = DataUtil.readGeometriesFromShapeFile(new File("data/regionalization/500rand.shp"));
		//List<double[]> samples = DataUtil.readSamplesFromShapeFile(new File("data/regionalization/500rand.shp"), new int[] {}, true);
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/redcap/Election/election2004.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/redcap/Election/election2004.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);

		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");
		//final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/500rand.ctg");
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/redcap/Election/election2004_Queen.ctg");
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		Grid2DHex<double[]> grid = new Grid2DHex<double[]>(4, 4);
		SomUtils.initRandom(grid, samples);
		log.debug("cluster: "+grid.getPositions().size() );
				
		{ 
			log.debug("training init som");
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter(fDist);

			SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5, 0.0), grid, bmuGetter);
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}
			
			try { 
				Drawer.geoDrawCluster(SomUtils.getBmuMapping(samples, grid, bmuGetter).values(), samples, geoms, new FileOutputStream("output/pre_geo.png"), true); 
			} catch(FileNotFoundException e) {
				e.printStackTrace(); 
			}
		}
		
		log.debug("training regio som");
		SimpleRegionBmuGetter bmuGetter = new SimpleRegionBmuGetter(cm, fDist);
		bmuGetter.getMap().put( grid.getPositions().iterator().next(), new HashSet<double[]>(samples) );
		
		SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5, 0.0), grid, bmuGetter);
		for (int t = 0; t < T_MAX; t++) {
			double[] x = samples.get(r.nextInt(samples.size()));
			som.train((double) t / T_MAX, x);
		}
				
		// trainining finished... fill map
		Set<double[]> all = new HashSet<double[]>();

		while (all.size() != samples.size()) {
			for (Set<double[]> s : bmuGetter.getMap().values())
				all.addAll(s);
			
			log.debug(all.size());

			for (double[] d1 : samples) {
				if (all.contains(d1))
					continue;

				for (double[] d2 : cm.get(d1))
					if (all.contains(d2)) {
						bmuGetter.getBmuPos(d1, grid);
						break;
					}
			}
		}

		if (all.size() != samples.size()) {
			log.error("Not all samples are mapped!!!");
			System.exit(1);
		}

		boolean cont = true;
		for (Set<double[]> s : bmuGetter.getMap().values())
			if (!RegionUtils.isContiugous(cm, s))
				cont = false;

		if (!cont) {
			log.error("Not contiguous!!!");
			System.exit(1);
		}
		
		try { 
			SomUtils.printUMatrix(grid, fDist, new FileOutputStream("output/umatrix.png"));
			SomUtils.printDMatrix(grid, fDist, new FileOutputStream("output/dmatrix.png"));
			Drawer.geoDrawCluster(bmuGetter.getMap().values(), samples, geoms, new FileOutputStream("output/final_geo.png"), true); 
		} catch(FileNotFoundException e) {
			e.printStackTrace(); 
		}
			 
		log.info("Heterogenity: " + RegionUtils.getHeterogenity(bmuGetter.getMap().values(), fa));
		log.debug("qe: "+SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples) );
				
		// new GeoPicker(samples, new String[]{"area","fips","bush","kerry","county","nader","total","bush_pct","kerry_pct","nader_pct"}, geoms, SomUtils.getBmuMapping(samples, grid, bmuGetter), grid, fDist);
	}
}
