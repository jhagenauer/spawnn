package regionalization.ga;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import spawnn.dist.EuclideanDist;
import spawnn.utils.DataUtils;
import spawnn.utils.Drawer;
import spawnn.utils.RegionUtils;

import com.vividsolutions.jts.geom.Geometry;

public class RegioGAIndividual2 implements GAIndividual {
		
	protected List<double[]> genome;
	protected int seedSize;
	final protected Map<double[], Set<double[]>> cm;
	protected final ClusterCostCalculator cc;
		
	protected List<Set<double[]>> regList;		
	protected double cost;
	
	public List<Integer> cuts = new ArrayList<Integer>();
	
	public RegioGAIndividual2( final List<double[]> genome, ClusterCostCalculator cc, final Map<double[], Set<double[]>> cm ) {
		this.cm = cm;		
		this.cc = cc;
		
		this.genome = new ArrayList<double[]>(genome);
		
		this.regList = new ArrayList<Set<double[]>>();
			
		cuts.add(0);	
		for( int i = 1; i < genome.size(); i++ ) {
			double[] d = genome.get(i);
			if( !cm.get( genome.get(i-1) ).contains(d) )  // start new cluster and safe old
				cuts.add(i);
		}
		cuts.add(genome.size());
				
		for( int i = 1; i < cuts.size(); i++ ) 
			regList.add( new HashSet<double[]>( genome.subList( cuts.get(i-1), cuts.get(i) ) ) );
				
		cost = 0;
		/*for( Set<double[]> s : regList )
			cost += cc.getCost(s);*/
		
		cost += regList.size();
	}

	public List<double[]> getGenome() {
		return genome;
	}
				
	public List<Set<double[]>> getRegionList() {
		return regList;
	}
	
	@Override
	public GAIndividual mutate() {
		return mutate(0, genome.size() );
	}
	
	@Override
	public GAIndividual recombine(GAIndividual mother) {
		return partiallyMatched(mother, 0, genome.size() );
	}
	
	/* TODO: Idea: partially matched mutation (muatet one cluster by pmx) and crossover (change multiple clusters) 
	 */
		
	private GAIndividual mutate( int start, int size ) {
		Random r = new Random();
		List<double[]> nGenome = new ArrayList<double[]>(genome);
		
		// exchange single element within length with one from complete genome
		for( int i = 0; i < size; i++ ) {
			if( r.nextDouble() < 1.0/size ) {
				int idxA = start + r.nextInt(size);
				int idxB = r.nextInt(nGenome.size());
						
				double[] valB = nGenome.set( idxB, nGenome.get(idxA) );
				nGenome.set(idxA, valB );	
			}
		}
			
		return new RegioGAIndividual2( nGenome, cc, cm );
	}
		
	public GAIndividual partiallyMatched(GAIndividual mother, int start, int length ) {
		Random r = new Random();
		List<double[]> mGenome = ((RegioGAIndividual2)mother).getGenome();
				
		int idxA = start + r.nextInt( length - 1 );
		int idxB = idxA + r.nextInt( start + length - idxA ) + 1;
								
		List<double[]> nGenome = new ArrayList<double[]>( genome.size() );
		for( int i = 0; i < genome.size(); i++ )
			nGenome.add(null);
							
		// keep between idxA and idxB
		Set<double[]> keep = new HashSet<double[]>();
		for( int i = idxA; i < idxB; i++ ) {
			nGenome.set(i, genome.get(i) );
			keep.add( genome.get(i) );
		}
							
		// fill with mother
		for( int i = 0; i < genome.size(); i++ )
			if( nGenome.get(i) == null && !keep.contains( mGenome.get(i) ) )
				nGenome.set(i, mGenome.get(i) );
						
		// fill rest with genome
		List<double[]> rest = new LinkedList<double[]>(genome);
		rest.removeAll(nGenome);
		for( int i = genome.size()-1; i >= 0; i-- )
			if( nGenome.get(i) == null )
				nGenome.set(i, rest.remove(0) );
					
		return new RegioGAIndividual2( nGenome, cc, cm );
	}
	
	@Override
	public int compareTo(GAIndividual o) {
		if( getValue() < o.getValue() )
			return -1;
		else if( getValue() > o.getValue() )
			return 1;
		else
			return 0;
	}

	@Override
	public double getValue() {
		return cost;
	}
	
	public static void main(String[] args) {
		List<Geometry> geoms = DataUtils.readGeometriesFromShapeFile(new File("data/regionalization/200rand.shp"));
		List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/regionalization/200rand.shp"), new int[] {}, true);
		
		int[] fa = new int[] { 7 };

		for (int i : fa)
			DataUtils.zScoreColumn(samples, i);
		
		final Map<double[], Set<double[]>> cm = RegionUtils.readContiguitiyMap(samples, "data/regionalization/200rand.ctg");
		
		for( int i = 0; i < 1; i++ ) {
			List<double[]> chromosome = new ArrayList<double[]>(samples);
			//Collections.shuffle(chromosome);
			RegioGAIndividual2 rgi = new RegioGAIndividual2( chromosome, new WCSSCostCalulator(new EuclideanDist(fa)), cm );
						
			try { 
				Drawer.geoDrawCluster( rgi.getRegionList(), samples, geoms, new FileOutputStream("output/rgi.png"), true ); 
			} catch(FileNotFoundException e) {
				e.printStackTrace(); 
			}	
			
			System.exit(1);
		}	
	}
}
