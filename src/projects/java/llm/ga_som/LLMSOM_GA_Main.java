package llm.ga_som;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Coordinate;

import cern.colt.Arrays;
import ga.CostCalculator;
import ga.GeneticAlgorithm;
import llm.LLMNG;
import llm.LLMSOM;
import llm.ga_ng.LLMNG_Individual;
import spawnn.dist.EuclideanDist;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.Grid2DHexToroid;
import spawnn.som.grid.GridPos;
import spawnn.som.utils.SomUtils;
import spawnn.utils.ColorBrewer;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class LLMSOM_GA_Main {

	private static Logger log = Logger.getLogger(LLMSOM_GA_Main.class);

	public static void main(String[] args) {
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/adep/dep.data.all.muni.shp"), true);
		
		List<double[]> samples = new ArrayList<>();
		for( int i = 0; i < sdf.samples.size(); i++ ) {
			
			double[] d = sdf.samples.get(i);
			if( d[0] < 0 )
				continue;
						
			Coordinate[] coord = sdf.geoms.get(i).getCentroid().getCoordinates();
			assert coord.length == 1;
			
			double[] nd = new double[12];
			nd[0] = d[2]; // green
			nd[1] = d[5]; // elderly
			nd[2] = d[6]; // unemp
			nd[3] = d[7]; // house
			nd[4] = d[8]; // gp
			nd[5] = d[9]+d[10]; // act := walk + cycl
			nd[6] = d[11]; // dens
			nd[7] = d[12]; // SMR
			nd[8] = d[13]; // nonw
			nd[9] = coord[0].x; // X
			nd[10] = coord[0].y; // y
			nd[11] = d[0]; // adep
						
			samples.add(nd);
		}
		String[] names = new String[]{"green","elderly","unemp","house","gp","act","dens","SMR","nonw","X","Y","adep"};
		
		//int[] fa = new int[]{5,7}; 
		int[] fa = new int[]{0,1,2,3,5,7,8}; 
		int[] ga = new int[]{9,10}; 
		int ta = 11;
					
		//DataUtils.transform(samples, new int[]{4,6,8}, Transform.log );
		
		DataUtils.transform(samples, fa, Transform.zScore);
		//DataUtils.transform(samples, ta, Transform.zScore);
			
		// LLMNG test
		boolean test_ng = false;
		if( test_ng ){
			log.debug("--- LLMNG ---");
			
			LLMNG_Individual.nrNeurons = 1;
			
			CostCalculator<LLMNG_Individual> cc2 = new llm.ga_ng.LLMNG_CV_CostCalculator(samples, fa, ta);
			//CostCalculator<LLMNG_Individual> cc2 = new llm.ga_ng.LLMNG_QE_CostCalculator(samples, fa, ta);
												
			for( LLMNG_Individual i : new LLMNG_Individual[]{
					new LLMNG_Individual("{lr1Final=1.0E-5, lr1Func=Power, lr1Init=0.3, lr2Final=1.0E-5, lr2Func=Power, lr2Init=0.1, mode=fritzke, nb1Final=1.0E-4, nb1Func=Power, nb1Init=2.0, nb2Final=1.0E-4, nb2Func=Linear, nb2Init=3.0, t_max=100000, w=1.0}"),
					new LLMNG_Individual("{lr1Final=0.01, lr1Func=Linear, lr1Init=0.8, lr2Final=1.0E-5, lr2Func=Power, lr2Init=0.1, mode=martinetz, nb1Final=1.0E-5, nb1Func=Power, nb1Init=0.6, nb2Final=1.0E-4, nb2Func=Power, nb2Init=0.8, t_max=100000, w=1.0}")
			} ) {
				log.debug(i);
				log.debug( cc2.getCost(i) );
				
				LLMNG llmng = i.train(samples, fa, ta);
				for( double[] n : llmng.getNeurons() ) {
					log.debug("n: "+Arrays.toString( strip(n,fa) )+", m: "+Arrays.toString( strip( llmng.matrix.get(n)[0], fa)  )+", o: "+llmng.output.get(n)[0] );
				}
			}
			//System.exit(1);
						
			GeneticAlgorithm.tournamentSize = 3;
			GeneticAlgorithm.elitist = true;
			GeneticAlgorithm.recombProb = 0.7;
			
			List<LLMNG_Individual> init = new ArrayList<>();
			while (init.size() < 100) {
				init.add(new LLMNG_Individual());
			}
			
			GeneticAlgorithm<LLMNG_Individual> gen = new GeneticAlgorithm<LLMNG_Individual>();
			LLMNG_Individual result = (LLMNG_Individual) gen.search(init, cc2);

			log.info("best:");
			log.info(cc2.getCost(result));
			log.info(result.iParam);
		}
		
		CostCalculator<LLMSOM_Individual> cc = new LLMSOM_CV_CostCalculator(samples, fa, ga, ta);
		
		for( LLMSOM_Individual i : new LLMSOM_Individual[]{
				new LLMSOM_Individual("{lr1Final=0.001, lr1Init=0.8, lr2Final=0.01, lr2Init=0.05, mode=martinetz, nb1Final=0.1, nb1Init=8.0, nb2Final=0.1, nb2Init=3.0, t_max=100000, type=toroid, w=2.0}")
		} ) {
			log.debug(i);
			log.debug( cc.getCost(i) );
			
			LLMSOM llmSOM = i.train(samples, fa, ga, ta);
			try {
				Grid2D<double[]> grid = (Grid2D<double[]>)llmSOM.getGrid();
				SomUtils.printHexUMat(grid, new EuclideanDist(fa), ColorBrewer.Blues, new FileOutputStream("output/"+llmSOM.hashCode()+"_umat_"+i.iParam+".png"));
				SomUtils.printDMatrix(grid, new EuclideanDist(fa), ColorBrewer.Blues, new FileOutputStream("output/"+llmSOM.hashCode()+"_dmat_"+i.iParam+".png"));
				
				for( int j : fa ) 
					SomUtils.printComponentPlane( grid, j, ColorBrewer.Blues, new FileOutputStream("output/"+llmSOM.hashCode()+"_comp_"+names[j]+"_"+i.iParam+".png"));
				
				Grid2D<double[]> coefGrid;
				if( grid instanceof Grid2DHexToroid )
					coefGrid = new Grid2DHexToroid<>(grid.getGridMap());
				else
					coefGrid = new Grid2DHex<>(grid.getGridMap());
				for( GridPos p : coefGrid.getPositions() ) 
					coefGrid.setPrototypeAt( p, llmSOM.matrix.get(p)[0] );
				
				for( int j = 0; j < fa.length; j++ )
					SomUtils.printComponentPlane( coefGrid, j, ColorBrewer.Blues, new FileOutputStream("output/"+llmSOM.hashCode()+"_coef_"+names[fa[j]]+"_"+i.iParam+".png"));
				
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
			
		GeneticAlgorithm.tournamentSize = 3;
		GeneticAlgorithm.elitist = true;
		GeneticAlgorithm.recombProb = 0.7;
		
		List<LLMSOM_Individual> init = new ArrayList<>();
		while (init.size() < 100) {
			init.add(new LLMSOM_Individual());
		}
		
		GeneticAlgorithm<LLMSOM_Individual> gen = new GeneticAlgorithm<LLMSOM_Individual>();
		LLMSOM_Individual result = (LLMSOM_Individual) gen.search(init, cc);

		log.info("best:");
		log.info(cc.getCost(result));
		log.info(result.iParam);
	}
	
	public static double[] strip( double[] d, int[] fa ) {
		double[] nd = new double[fa.length];
		for( int i = 0; i < fa.length; i++ )
			nd[i] = d[fa[i]];
		return nd;
	}
}
