#include "input_recording/inspect.h"
#include <iostream>
int main(int argc,char** argv){
    try{
        c3x_inputs::require(argc==3||argc==4,"usage: inspect_inputs INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY [--allow-prefix]");
        bool prefix=argc==4; c3x_inputs::require(!prefix||std::string(argv[3])=="--allow-prefix","unknown inspection option");
        auto output=std::filesystem::path(argv[2]);c3x_inputs::require(!std::filesystem::exists(output)&&std::filesystem::create_directories(output),"inspection output must be new");
        std::ofstream timeline(output/"timeline.jsonl");c3x_inputs::require(bool(timeline),"cannot create input timeline");
        c3x_inputs::InputInspection inspection;inspection.read(argv[1],timeline);timeline.close();
        std::ofstream report(output/"report.json");inspection.report(report);report.close();inspection.report(std::cout);
        return inspection.complete||(prefix&&inspection.verified)?0:1;
    }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
