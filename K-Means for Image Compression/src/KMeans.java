import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;
 

public class KMeans {
	
    public static void main(String [] args){
	if (args.length < 3){
	    System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
	    return;
	}
	try{
	    BufferedImage originalImage = ImageIO.read(new File(args[0]));
	    int k=Integer.parseInt(args[1]);
	    BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
	    ImageIO.write(kmeansJpg, "jpg", new File(args[2])); 
	    
	}catch(IOException e){
	    System.out.println(e.getMessage());
	}	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
    	int w=originalImage.getWidth();
    	int h=originalImage.getHeight();
    	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
    	Graphics2D g = kmeansImage.createGraphics();
    	g.drawImage(originalImage, 0, 0, w,h , null);
	
    	// Read rgb values from the image
    	int[] rgb=new int[w*h];
    	int count=0;
    	for(int i=0;i<w;i++){
    		for(int j=0;j<h;j++){
    			rgb[count++]=kmeansImage.getRGB(i,j);
    			
    		}
    	}
   
    	// Call kmeans algorithm: update the rgb values
    	rgb = kmeans(rgb,k);
	
    	// Write the new rgb values to the image
    	count=0;
    	for(int i=0;i<w;i++){
    		for(int j=0;j<h;j++){
    			kmeansImage.setRGB(i,j,rgb[count++]);
    		}
    	}
	
    	System.out.println("Image has compressed!");
    	return kmeansImage;
    }
    
    //Get the initially random means
	private static int[] getRandom(int[] rgb,int k){
		int[] means = new int[k];
		Random rand = new Random();
		int random;
		for(int i=0;i<k;i++){
			random = rand.nextInt(rgb.length - 1);
			means[i] = random;
		}
		return means;
	}
	
	//Check if current means are the same as previous means
	private static boolean ifConverge(int[] preMeans, int[] currMeans, int iterations){
		int maxIteration = 50;
		boolean ifConverge = false;		
		if(iterations > maxIteration){
			return false;	
		}
		else{
			for(int i=0;i < currMeans.length;i++){
				if(currMeans[i] != preMeans[i])
				{
					ifConverge = true;
					break;
				}
			}
		}				
		return ifConverge;
	}
	
	//For each pixel, find the most closest mean, then label them
	private static int[] findMinDist(int[] data, int[] means){
		int[] labels = new int[data.length];
		for(int i=0;i<data.length;i++){		
			int minDist= Integer.MAX_VALUE;
			int minMean=0;  		
			for(int j=0;j<means.length;j++){
				int dist = Math.abs(data[i] - means[j]);
				if(dist< minDist){
					minDist = dist;
					minMean = j;
				}					
			}			
			labels[i] = minMean;
		}
		return labels;
	}

	//Calculate the means for k labels
	private static int[] getMeans(int[] data, int[] labels, int k){
		int means[] = new int[k];		
		for(int i=0;i<k;i++){			
			int countCluster = 0;
			long sumOfCluster = 0;
			for(int j=0;j<data.length;j++){
				if(labels[j] == i){
					countCluster++;
					sumOfCluster += data[j];
				}
			}
			if(countCluster == 0)
				means[i] = 0;
			else
				means[i] = (int)(sumOfCluster/countCluster);				
		}
		return means;
	}
	
    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static int[] kmeans(int[] rgb, int k){
 	
		int iterations = 0;
		int[] prevMeans = new int[k];
		int[] currMeans = getRandom(rgb, k);
		int[] labels = new int[rgb.length];
				
		while(ifConverge(prevMeans, currMeans, iterations)){
			System.out.println("Iteration : " + iterations);
			prevMeans = currMeans;
			iterations++;			
			labels = findMinDist(rgb,currMeans);
			currMeans = getMeans(rgb, labels, k);
		}
			
		for(int i=0;i<labels.length;i++){
			int index = labels[i];
			rgb[i] = (int)currMeans[index];
		}

		return rgb;
    }

}