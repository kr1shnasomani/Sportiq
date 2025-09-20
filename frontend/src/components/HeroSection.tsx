import { Button } from "@/components/ui/button";
import { ArrowRight, Play, Target, TrendingUp } from "lucide-react";
import heroImage from "@/assets/hero-tennis-analytics.jpg";
const HeroSection = () => {
  const scrollToUpload = () => {
    document.getElementById('upload')?.scrollIntoView({
      behavior: 'smooth'
    });
  };
  return <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background/95 to-background/90">
        <img src={heroImage} alt="Tennis court with AI analytics overlay" className="w-full h-full object-cover opacity-20" />
        <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-background/20" />
      </div>

      {/* Grid Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
        backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--primary)) 1px, transparent 0)',
        backgroundSize: '20px 20px'
      }} />
      </div>

      <div className="container relative z-10">
        <div className="mx-auto max-w-4xl text-center space-y-8">
          {/* Badge */}
          

          {/* Main Heading */}
          <div className="space-y-4">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-foreground via-foreground to-primary bg-clip-text text-transparent">Sportiq</span>
              <br />
              
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">Player, ball tracking and court detection map for Tennis using computer vision, Mediapipe and TrackNet</p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-4 text-sm">
            <div className="flex items-center space-x-2 bg-background/50 border border-border/50 rounded-full px-4 py-2">
              <Target className="w-4 h-4 text-primary" />
              <span>Player Tracking</span>
            </div>
            <div className="flex items-center space-x-2 bg-background/50 border border-border/50 rounded-full px-4 py-2">
              <TrendingUp className="w-4 h-4 text-tech-blue" />
              <span>Ball Tracking</span>
            </div>
            <div className="flex items-center space-x-2 bg-background/50 border border-border/50 rounded-full px-4 py-2">
              <Play className="w-4 h-4 text-success-green" />
              <span>Court Detection</span>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button size="lg" onClick={scrollToUpload} className="group px-8 py-6 text-lg bg-gradient-to-r from-primary to-tennis-green hover:from-primary/90 hover:to-tennis-green/90">
              Upload Your Video
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </div>

          {/* Stats */}
          
        </div>
      </div>
    </section>;
};
export default HeroSection;