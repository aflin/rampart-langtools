/* llama_gen_macos.mm — macOS-only helper.
 *
 * Flip the Cocoa/Foundation runtime into multithreaded mode. Foundation's
 * internals (and the dispatch_once-based lazy initialization used throughout
 * Metal) are only guaranteed thread-safe once [NSThread isMultiThreaded] is YES,
 * which happens the first time an NSThread is detached. rampart creates its
 * threads with raw pthread_create, so without this the runtime can stay in
 * single-threaded mode and crash when our dedicated inference thread runs Metal
 * concurrently with a JS thread's event loop also touching Foundation.
 */
#import <Foundation/Foundation.h>

@interface RPGenMTPrimer : NSObject
+ (void)noop:(id)arg;
@end
@implementation RPGenMTPrimer
+ (void)noop:(id)arg { (void)arg; }
@end

extern "C" void lg_cocoa_make_multithreaded(void) {
    if (![NSThread isMultiThreaded]) {
        /* detach a thread that immediately exits — permanently flips the runtime
         * to multithreaded mode (isMultiThreaded stays YES afterwards). */
        [NSThread detachNewThreadSelector:@selector(noop:)
                                 toTarget:[RPGenMTPrimer class]
                               withObject:nil];
    }
}
